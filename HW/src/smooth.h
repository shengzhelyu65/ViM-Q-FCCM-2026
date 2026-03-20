#ifndef __INT_SMOOTH_H__
#define __INT_SMOOTH_H__

#include "common.h"
#include "linear_block.h"
#include <hls_stream.h>

constexpr unsigned int SMOOTH_SCALES_PER_WORD = AXI_XFER_BIT_WIDTH / 32; // 8 elements per 256-bit word

void load_smooth_scales(
    wt_linear_ss_block_t dst[],
    const wt_wide_t src[],
    unsigned int in_dim
)
{
    #pragma HLS inline off
    
    static_assert(SMOOTH_SCALES_PER_WORD == FEATURE_BLOCK_SIZE, 
                  "SMOOTH_SCALES_PER_WORD must equal FEATURE_BLOCK_SIZE for direct mapping");
    
    unsigned int num_blocks = ceildiv(in_dim, FEATURE_BLOCK_SIZE);
    unsigned int num_words = ceildiv(in_dim, SMOOTH_SCALES_PER_WORD);

    FOR_EACH(i, num_words)
    {
        #pragma HLS pipeline II=1
        wt_wide_t word = src[i];
        
        FOR_EACH(j, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            unsigned int idx = i * SMOOTH_SCALES_PER_WORD + j;
            if (idx < in_dim) {
                ap_uint<32> elem_bits = word.range(j * 32 + 31, j * 32);
                dst[i][j].range(31, 0) = elem_bits;
            } else {
                dst[i][j] = (wt_linear_ss_t)0;
            }
        }
    }
}

void read_smooth_in_stream(
    hls::stream<fm_block_t>& in_stream,
    const fm_block_t src[],
    unsigned int in_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    static_assert(LINEAR_BLOCK_SIZE % FEATURE_BLOCK_SIZE == 0, "LINEAR_BLOCK_SIZE must be a multiple of FEATURE_BLOCK_SIZE");
    
    constexpr unsigned int num_blocks = LINEAR_BLOCK_SIZE / FEATURE_BLOCK_SIZE;
    constexpr unsigned int last_block = num_blocks - 1;
    unsigned int next_block = 0;

    fm_block_t blocks[num_blocks];
    #pragma HLS array_partition variable=blocks complete

    unsigned int iters = num_patches * ceildiv(in_dim, FEATURE_BLOCK_SIZE);

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=1 max=(NUM_PATCHES*ceildiv(D_INNER,FEATURE_BLOCK_SIZE))

        unsigned int block = next_block;
        next_block = (block == last_block) ? 0 : block + 1;

        blocks[block] = src[i];

        if (block == last_block)
        {
            #pragma HLS occurrence cycle=num_blocks

            FOR_EACH(j, num_blocks)
            {
                #pragma HLS unroll
                in_stream << blocks[j];
            }
        }
    }
}

void compute_smooth_on_stream(
    hls::stream<fm_block_t>& out_stream,
    hls::stream<fm_block_t>& in_stream,
    const wt_linear_ss_block_t smooth_scales[],
    unsigned int in_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    static_assert(LINEAR_BLOCK_SIZE % FEATURE_BLOCK_SIZE == 0, "LINEAR_BLOCK_SIZE must be a multiple of FEATURE_BLOCK_SIZE");
    
    constexpr unsigned int num_blocks = LINEAR_BLOCK_SIZE / FEATURE_BLOCK_SIZE;
    unsigned int in_dim_iters = ceildiv(in_dim, LINEAR_BLOCK_SIZE);

    FOR_EACH(patch, num_patches)
    {
        FOR_EACH(in_dim_block, in_dim_iters)
        {
            #pragma HLS pipeline II=num_blocks
            #pragma HLS loop_tripcount min=ceildiv(D_INNER,LINEAR_BLOCK_SIZE) max=ceildiv(D_INNER,LINEAR_BLOCK_SIZE)

            fm_block_t blocks[num_blocks];
            #pragma HLS array_partition variable=blocks complete

            FOR_EACH(j, num_blocks)
            {
                #pragma HLS unroll
                in_stream >> blocks[j];
            }

            FOR_EACH(j, num_blocks)
            {
                #pragma HLS unroll
                
                unsigned int global_block = in_dim_block * num_blocks + j;
                fm_block_t out_val;
                
                #pragma HLS dependence variable=smooth_scales inter false
                
                FOR_EACH(offset, FEATURE_BLOCK_SIZE)
                {
                    #pragma HLS unroll
                    out_val[offset] = blocks[j][offset] * smooth_scales[global_block][offset];
                }
                
                out_stream << out_val;
            }
        }
    }
}

void write_smooth_out_stream(
    fm_block_t dst[],
    hls::stream<fm_block_t>& out_stream,
    unsigned int in_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    static_assert(LINEAR_BLOCK_SIZE % FEATURE_BLOCK_SIZE == 0, "LINEAR_BLOCK_SIZE must be a multiple of FEATURE_BLOCK_SIZE");
    
    constexpr unsigned int num_blocks = LINEAR_BLOCK_SIZE / FEATURE_BLOCK_SIZE;
    constexpr unsigned int last_block = num_blocks - 1;
    unsigned int next_block = 0;

    fm_block_t blocks[num_blocks];
    #pragma HLS array_partition variable=blocks complete

    unsigned int iters = num_patches * ceildiv(in_dim, FEATURE_BLOCK_SIZE);

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=1 max=(NUM_PATCHES*ceildiv(D_INNER,FEATURE_BLOCK_SIZE))

        unsigned int block = next_block;
        next_block = (block == last_block) ? 0 : block + 1;

        if (block == 0)
        {
            #pragma HLS occurrence cycle=num_blocks

            FOR_EACH(j, num_blocks)
            {
                #pragma HLS unroll
                out_stream >> blocks[j];
            }
        }

        dst[i] = blocks[block];
    }
}

static void smooth_layer_impl(
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_linear_ss_block_t smooth_scales[],
    unsigned int in_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    #pragma HLS dataflow

    hls::stream<fm_block_t> in_stream("smooth_in_stream");
    #pragma HLS stream variable=in_stream depth=2
    hls::stream<fm_block_t> out_stream("smooth_out_stream");
    #pragma HLS stream variable=out_stream depth=2

    read_smooth_in_stream(in_stream, src, in_dim, num_patches);
    compute_smooth_on_stream(out_stream, in_stream, smooth_scales, in_dim, num_patches);
    write_smooth_out_stream(dst, out_stream, in_dim, num_patches);
}

void compute_smooth(
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_wide_t smooth_scales_base[],
    unsigned int in_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    static wt_linear_ss_block_t smooth_scales[ceildiv(MAX_D_INNER, FEATURE_BLOCK_SIZE)];

    load_smooth_scales(smooth_scales, smooth_scales_base, in_dim);
    smooth_layer_impl(dst, src, smooth_scales, in_dim, num_patches);
}

#endif
