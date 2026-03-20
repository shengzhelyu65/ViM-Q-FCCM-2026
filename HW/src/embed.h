#ifndef __INT_EMBED_H__
#define __INT_EMBED_H__

#include "common.h"
#include <hls_stream.h>

// ============================================================================
// Parameters
// ============================================================================

constexpr unsigned int EMBED_PAR_CHANNELS = 32;

// ============================================================================
// Implementation Functions
// ============================================================================

void patch_embed_impl(
    fm_block_t *out,
    const wt_wide_t *image,
    const wt_wide_t *weights,
    const wt_wide_t *bias,
    const wt_wide_t *pos_embed,
    const wt_wide_t *cls_token,
    const unsigned int dim,
    const unsigned int numPatches,
    const unsigned int imgHeight,
    const unsigned int imgWidth
) {
    #pragma HLS inline off

    // 1. Local Buffers
    // Weights and Bias are cached for the entire image processing.
    constexpr unsigned int WT_PER_WORD = 256 / 16;
    // Optimize: Use 16-bit storage to avoid BRAM width fragmentation
    static wt_patch_embed_t W_local[D_MODEL][3][PATCH_HEIGHT * PATCH_WIDTH];
    #pragma HLS bind_storage variable=W_local type=ram_2p impl=bram
    #pragma HLS array_partition variable=W_local cyclic factor=32 dim=1     // Parallel channels

    static wt_patch_bias_t B_local[D_MODEL];
    #pragma HLS array_partition variable=B_local cyclic factor=32

    static fm_block_t CLS_local[ceildiv(D_MODEL, FEATURE_BLOCK_SIZE)];
    #pragma HLS array_partition variable=CLS_local complete

    const unsigned int num_feature_blocks = dim / FEATURE_BLOCK_SIZE;
    const unsigned int patches_per_row = imgWidth / PATCH_WIDTH;

    // 2. Load weights, bias, and CLS token into local buffers
    // Use wide loading for better performance
    
    // Bias: 192 elements (16-bit each) -> 12 words
    constexpr unsigned int BIAS_PER_WORD = 256 / 16; // 16 elements per word
    load_bias: for(int i=0; i < ceildiv(dim, BIAS_PER_WORD); i++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=12 max=12
        wt_wide_t word = bias[i];
        for(int j=0; j < BIAS_PER_WORD; j++) {
            unsigned int idx = i * BIAS_PER_WORD + j;
            B_local[idx].range(15, 0) = word.range(j * 16 + 15, j * 16);
        }
    }

    // CLS Token: dim elements (32-bit each) -> dim/8 blocks (8*32=256-bit each)
    load_cls: for(int i=0; i < num_feature_blocks; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=24 max=24
        wt_wide_t word = cls_token[i];
        fm_block_t v;
        for(int j=0; j < 8; j++) {
            #pragma HLS unroll
            v[j].range(31, 0) = word.range(j * 32 + 31, j * 32);
        }
        CLS_local[i] = v;
    }

    // Weights: oc * 3 * 16 * 16 elements (16-bit each)
    // Each 16-bit weight, 16 elements per 256-bit word.
    load_weights_flat: for(int i=0; i < dim * 3 * (PATCH_HEIGHT * PATCH_WIDTH) / WT_PER_WORD; i++) {
        #pragma HLS pipeline II=8
        #pragma HLS loop_tripcount min=9216 max=9216
        wt_wide_t word = weights[i];
        
        unsigned int total_p_words = 3 * (PATCH_HEIGHT * PATCH_WIDTH) / WT_PER_WORD;
        unsigned int oc = i / total_p_words;
        unsigned int rem = i % total_p_words;
        unsigned int ic = rem / ((PATCH_HEIGHT * PATCH_WIDTH) / WT_PER_WORD);
        unsigned int p_base = (rem % ((PATCH_HEIGHT * PATCH_WIDTH) / WT_PER_WORD)) * WT_PER_WORD;
        
        for (int k = 0; k < WT_PER_WORD; k++) {
            W_local[oc][ic][p_base + k].range(15, 0) = word.range(k * 16 + 15, k * 16);
        }
    }

    unsigned int img_p_idx = 0;
    
    patches_loop: for (unsigned int p = 0; p < numPatches; p++) {
        #pragma HLS pipeline off
        #pragma HLS loop_tripcount min=197 max=197

        if (p == numPatches / 2) {
            // CLS Token: Just add pos_embed and write out
            cls_loop: for(int b=0; b < num_feature_blocks; b++) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=24 max=24
                fm_block_t cv = CLS_local[b];
                wt_wide_t pe_word = pos_embed[p * num_feature_blocks + b];
                fm_block_t pv;
                for(int i=0; i<8; i++) {
                    #pragma HLS unroll
                    pv[i].range(31, 0) = pe_word.range(i * 32 + 31, i * 32);
                }
                
                fm_block_t res;
                for(int i=0; i<8; i++) {
                    #pragma HLS unroll
                    res[i] = (fm_t)cv[i] + pv[i];
                }
                out[p * num_feature_blocks + b] = res;
            }
        } else {
            // Image Patch
            unsigned int r_base = (img_p_idx / patches_per_row) * PATCH_HEIGHT;
            unsigned int c_base = (img_p_idx % patches_per_row) * PATCH_WIDTH;
            img_p_idx++;

            // Image is pixel_t (32-bit fixed). 8 pixels per 256-bit word.
            constexpr unsigned int PX_PER_WORD = 256 / 32;

            // Local buffer for one patch to avoid repeated image reads
            // Optimize: Use 32-bit storage to save BRAMs
            pixel_t patch_buf[3 * PATCH_HEIGHT * PATCH_WIDTH];
            #pragma HLS array_partition variable=patch_buf cyclic factor=8 dim=1

            unsigned int img_offset_base = (r_base * imgWidth + c_base) / PX_PER_WORD;
            unsigned int img_stride = imgWidth / PX_PER_WORD;
            unsigned int ch_stride = (imgHeight * imgWidth) / PX_PER_WORD;

            // Flatten load_img_patch for better burst inference
            load_img_patch: for(int i=0; i < 3 * (PATCH_HEIGHT * PATCH_WIDTH) / PX_PER_WORD; i++) {
                #pragma HLS pipeline II=4
                #pragma HLS loop_tripcount min=96 max=96
                
                unsigned int ic = i / ((PATCH_HEIGHT * PATCH_WIDTH) / PX_PER_WORD);
                unsigned int rem = i % ((PATCH_HEIGHT * PATCH_WIDTH) / PX_PER_WORD);
                unsigned int ph = rem / (PATCH_WIDTH / PX_PER_WORD);
                unsigned int pw_block = rem % (PATCH_WIDTH / PX_PER_WORD);
                
                unsigned int row_offset = ph * img_stride + pw_block;
                wt_wide_t word = image[ic * ch_stride + img_offset_base + row_offset];
                
                unsigned int base_idx = i * PX_PER_WORD;
                for(int k=0; k < PX_PER_WORD; k++) {
                    patch_buf[base_idx + k].range(31, 0) = word.range(k * 32 + 31, k * 32);
                }
            }

            // High-performance convolution: 
            // Process EMBED_PAR_CHANNELS output channels in parallel.
            conv_blocks: for(int b_base = 0; b_base < dim; b_base += EMBED_PAR_CHANNELS) {
                #pragma HLS pipeline off
                #pragma HLS loop_tripcount min=6 max=6
                
                fm_t acc[EMBED_PAR_CHANNELS];
                #pragma HLS array_partition variable=acc complete

                // Initialize with bias and pos_embed
                init_acc: for(int b_sub=0; b_sub < EMBED_PAR_CHANNELS / 8; b_sub++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=4 max=4
                    unsigned int b_idx = (b_base / 8) + b_sub;
                    wt_wide_t pe_word = pos_embed[p * num_feature_blocks + b_idx];
                    for(int i=0; i < 8; i++) {
                        #pragma HLS unroll
                        unsigned int oc = b_base + b_sub * 8 + i;
                        fm_t pe_val;
                        pe_val.range(31, 0) = pe_word.range(i * 32 + 31, i * 32);
                        acc[b_sub * 8 + i] = (fm_t)B_local[oc] + pe_val;
                    }
                }

                // Pipelined convolution for EMBED_PAR_CHANNELS channels
                conv_loop: for(int p_idx=0; p_idx < PATCH_HEIGHT * PATCH_WIDTH; p_idx++) {
                    #pragma HLS pipeline II=1
                    
                    pixel_t s0, s1, s2;
                    // patch_buf is now flat [3 * pixels_per_patch]
                    unsigned int pixels_per_patch = PATCH_HEIGHT * PATCH_WIDTH;
                    s0 = patch_buf[0 * pixels_per_patch + p_idx];
                    s1 = patch_buf[1 * pixels_per_patch + p_idx];
                    s2 = patch_buf[2 * pixels_per_patch + p_idx];

                    for(int i=0; i < EMBED_PAR_CHANNELS; i++) {
                        #pragma HLS unroll
                        unsigned int oc = b_base + i;
                        wt_patch_embed_t w0, w1, w2;
                        w0 = W_local[oc][0][p_idx];
                        w1 = W_local[oc][1][p_idx];
                        w2 = W_local[oc][2][p_idx];

                        acc[i] += (fm_t)(s0 * w0);
                        acc[i] += (fm_t)(s1 * w1);
                        acc[i] += (fm_t)(s2 * w2);
                    }
                }

                // Write out the result
                write_out: for(int b_sub=0; b_sub < EMBED_PAR_CHANNELS / 8; b_sub++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=4 max=4
                    fm_block_t v;
                    for(int i=0; i < 8; i++) {
                        #pragma HLS unroll
                        v[i] = acc[b_sub * 8 + i];
                    }
                    out[p * num_feature_blocks + (b_base / 8) + b_sub] = v;
                }
            }
        }
    }
}

#endif
