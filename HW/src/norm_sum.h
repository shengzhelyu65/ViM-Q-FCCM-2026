#ifndef __INT_NORM_SUM_H__
#define __INT_NORM_SUM_H__

#include "common.h"
#include <hls_math.h>
#include <hls_stream.h>

// ============================================================================
// Type Definitions
// ============================================================================

typedef hls::vector<wt_norm_t, FEATURE_BLOCK_SIZE> wt_norm_block_t;

// Mode definitions
enum NormSumMode {
    NORM_SUM_BOTH = 0,      // norm(a + b)
    NORM_SUM_NORM_ONLY = 1, // norm(a)
    NORM_SUM_ADD_ONLY = 2,  // a + b
    NORM_SUM_DIV2_ONLY = 3  // (a + b) / 2
};

// ============================================================================
// Helper Functions
// ============================================================================

void load_norm_weights(
    wt_norm_block_t dst[],
    const wt_wide_t src[],
    unsigned int model_dim
)
{
    #pragma HLS inline off
    
    unsigned int num_blocks = ceildiv(model_dim, FEATURE_BLOCK_SIZE);
    unsigned int num_words = ceildiv(model_dim, 16U);
    
    FOR_EACH(word_idx, num_words)
    {
        #pragma HLS pipeline II=2
        wt_wide_t word = src[word_idx];
        
        FOR_EACH(b, 2)
        {
            #pragma HLS unroll
            unsigned int block_idx = word_idx * 2 + b;
            if (block_idx < num_blocks) {
                unsigned int elem_offset = b * FEATURE_BLOCK_SIZE;
                FOR_EACH(j, FEATURE_BLOCK_SIZE)
                {
                    #pragma HLS unroll
                    unsigned int idx = block_idx * FEATURE_BLOCK_SIZE + j;
                    if (idx < model_dim) {
                        unsigned int elem_idx = elem_offset + j;
                        ap_uint<16> elem_bits = word.range(elem_idx * 16 + 15, elem_idx * 16);
                        dst[block_idx][j].range(15, 0) = elem_bits;
                    } else {
                        dst[block_idx][j] = (wt_norm_t)0;
                    }
                }
            }
        }
    }
}

void read_norm_sum_in_stream(
    hls::stream<fm_block_t>& in_stream,
    const fm_block_t src_a[],
    const fm_block_t src_b[],
    unsigned int model_dim,
    unsigned int num_patches,
    NormSumMode mode
)
{
    #pragma HLS inline off
    unsigned int num_blocks = ceildiv(model_dim, FEATURE_BLOCK_SIZE);
    unsigned int total_blocks = num_patches * num_blocks;

    FOR_EACH(i, total_blocks)
    {
        #pragma HLS pipeline II=1
        fm_block_t a = src_a[i];
        fm_block_t b = (mode == NORM_SUM_NORM_ONLY) ? (fm_block_t)0 : src_b[i];
        fm_block_t res;
        
        FOR_EACH(j, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            fm_t sum = a[j] + b[j];
            if (mode == NORM_SUM_DIV2_ONLY) {
                res[j] = sum * (fm_t)0.5;
            } else {
                res[j] = sum;
            }
        }
        in_stream.write(res);
    }
}

void compute_norm_on_stream(
    hls::stream<fm_block_t>& out_stream,
    hls::stream<fm_block_t>& in_stream,
    const wt_norm_block_t weights[],
    unsigned int model_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    
    fm_block_t patch_buffer[NUM_FEATURE_BLOCKS];
    #pragma HLS array_partition variable=patch_buffer complete dim=2

    unsigned int num_blocks = ceildiv(model_dim, FEATURE_BLOCK_SIZE);
    fm_t inv_model_dim = fm_t(1.0) / (fm_t)model_dim;

    FOR_EACH(patch, num_patches)
    {
        #pragma HLS loop_tripcount min=NUM_PATCHES max=NUM_PATCHES
        
        fm_t sum_sq = 0.00001; // epsilon

        // Step 1: Accumulate squares and buffer patch
        FOR_EACH(block, num_blocks)
        {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=NUM_FEATURE_BLOCKS max=NUM_FEATURE_BLOCKS
            fm_block_t val = in_stream.read();
            patch_buffer[block] = val;
            
            fm_t block_sum_sq = 0;
            FOR_EACH(i, FEATURE_BLOCK_SIZE)
            {
                #pragma HLS unroll
                fm_t v = val[i];
                block_sum_sq += v * v * inv_model_dim;
            }
            sum_sq += block_sum_sq;
        }
        
        // Step 2: Compute RMS inverse
        fm_t rms_inv = fm_t(1) / hls::sqrt(sum_sq);
        
        // Step 3: Apply weights and send to stream
        FOR_EACH(block, num_blocks)
        {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=NUM_FEATURE_BLOCKS max=NUM_FEATURE_BLOCKS
            fm_block_t val = patch_buffer[block];
            wt_norm_block_t w = weights[block];
            fm_block_t out_val;
            
            FOR_EACH(i, FEATURE_BLOCK_SIZE)
            {
                #pragma HLS unroll
                out_val[i] = val[i] * rms_inv * w[i];
            }
            out_stream.write(out_val);
        }
    }
}

void compute_norm_sum_on_stream(
    hls::stream<fm_block_t>& out_stream,
    hls::stream<fm_block_t>& in_stream,
    const wt_norm_block_t weights[],
    unsigned int model_dim,
    unsigned int num_patches,
    NormSumMode mode
)
{
    #pragma HLS inline off
    
    if (mode == NORM_SUM_ADD_ONLY || mode == NORM_SUM_DIV2_ONLY) {
        unsigned int num_blocks = ceildiv(model_dim, FEATURE_BLOCK_SIZE);
        unsigned int total_blocks = num_patches * num_blocks;
        FOR_EACH(i, total_blocks) {
            #pragma HLS pipeline II=1
            out_stream.write(in_stream.read());
        }
    } else {
        compute_norm_on_stream(out_stream, in_stream, weights, model_dim, num_patches);
    }
}

void write_norm_sum_out_stream(
    fm_block_t dst[],
    hls::stream<fm_block_t>& out_stream,
    unsigned int model_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    unsigned int num_blocks = ceildiv(model_dim, FEATURE_BLOCK_SIZE);
    unsigned int total_blocks = num_patches * num_blocks;

    FOR_EACH(i, total_blocks)
    {
        #pragma HLS pipeline II=1
        dst[i] = out_stream.read();
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

void norm_sum_impl(
    fm_block_t dst[],
    const fm_block_t src_a[],
    const fm_block_t src_b[],
    const wt_wide_t weights_base[],
    unsigned int model_dim,
    unsigned int num_patches,
    NormSumMode mode
)
{
    #pragma HLS inline off

    static wt_norm_block_t weights[NUM_FEATURE_BLOCKS];
    #pragma HLS array_partition variable=weights complete dim=2
    
    #pragma HLS dataflow
    
    hls::stream<fm_block_t> in_stream("in_stream");
    #pragma HLS stream variable=in_stream depth=2
    
    hls::stream<fm_block_t> out_stream("out_stream");
    #pragma HLS stream variable=out_stream depth=2

    load_norm_weights(weights, weights_base, model_dim);
    read_norm_sum_in_stream(in_stream, src_a, src_b, model_dim, num_patches, mode);
    compute_norm_sum_on_stream(out_stream, in_stream, weights, model_dim, num_patches, mode);
    write_norm_sum_out_stream(dst, out_stream, model_dim, num_patches);
}

#endif // __INT_NORM_SUM_H__
