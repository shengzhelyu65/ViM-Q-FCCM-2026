#ifndef __INT_PATCH_OPS_H__
#define __INT_PATCH_OPS_H__

#include "common.h"
#include <hls_stream.h>

// Mode definitions
enum PatchOpMode {
    PATCH_OP_FLIP = 0,      // Flip patch order
    PATCH_OP_LOAD_CLS = 1   // Load CLS token
};

static void patch_ops_flip_impl(
    fm_block_t dst[],
    const fm_block_t src[],
    unsigned int num_patches,
    unsigned int inner_dim
)
{
    #pragma HLS inline off
    
    unsigned int num_blocks_inner = ceildiv(inner_dim, FEATURE_BLOCK_SIZE);
    unsigned int num_blocks = num_patches * num_blocks_inner;
    unsigned int last_block = num_blocks_inner - 1;
    
    unsigned int next_patch = 0;
    unsigned int next_block = 0;
    
    FOR_EACH(i, num_blocks)
    {
        #pragma HLS pipeline II=1
        
        unsigned int patch = next_patch;
        unsigned int block = next_block;
        
        // Update counters
        next_block = (block == last_block) ? 0 : block + 1;
        next_patch = (block == last_block) ? patch + 1 : patch;
        
        unsigned int src_patch = num_patches - 1 - patch;
        unsigned int src_idx = src_patch * num_blocks_inner + block;
        dst[i] = src[src_idx];
    }
}

// ============================================================================
// Streaming implementation for LOAD_CLS_TOKEN
// ============================================================================

void read_patch_cls_stream(
    hls::stream<fm_block_t>& in_stream,
    const fm_block_t src[],
    unsigned int num_patches,
    unsigned int cls_token_idx,
    unsigned int model_dim
)
{
    #pragma HLS inline off
    
    unsigned int num_blocks = ceildiv(model_dim, FEATURE_BLOCK_SIZE);
    unsigned int cls_patch_start = cls_token_idx * num_blocks;
    
    FOR_EACH(b, num_blocks)
    {
        #pragma HLS pipeline II=1
        in_stream.write(src[cls_patch_start + b]);
    }
}

void process_patch_cls_stream(
    hls::stream<fm_block_t>& in_stream,
    hls::stream<fm_block_t>& out_stream,
    unsigned int num_patches,
    unsigned int model_dim
)
{
    #pragma HLS inline off
    
    unsigned int num_blocks = ceildiv(model_dim, FEATURE_BLOCK_SIZE);
    
    FOR_EACH(b, num_blocks)
    {
        #pragma HLS pipeline II=1
        out_stream.write(in_stream.read());
    }
}

void write_patch_cls_stream(
    fm_block_t dst[],
    hls::stream<fm_block_t>& out_stream,
    unsigned int num_patches,
    unsigned int model_dim
)
{
    #pragma HLS inline off
    
    unsigned int num_blocks = ceildiv(model_dim, FEATURE_BLOCK_SIZE);
    
    FOR_EACH(b, num_blocks)
    {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=NUM_FEATURE_BLOCKS max=NUM_FEATURE_BLOCKS
        dst[b] = out_stream.read();
    }
}

static void patch_ops_load_cls_impl(
    fm_block_t dst[],
    const fm_block_t src[],
    unsigned int num_patches,
    unsigned int cls_token_idx,
    unsigned int model_dim
)
{
    #pragma HLS inline off
    #pragma HLS dataflow
    
    hls::stream<fm_block_t> in_stream("patch_ops_cls_in_stream");
    #pragma HLS stream variable=in_stream depth=2
    
    hls::stream<fm_block_t> out_stream("patch_ops_cls_out_stream");
    #pragma HLS stream variable=out_stream depth=2
    
    read_patch_cls_stream(in_stream, src, num_patches, cls_token_idx, model_dim);
    process_patch_cls_stream(in_stream, out_stream, num_patches, model_dim);
    write_patch_cls_stream(dst, out_stream, num_patches, model_dim);
}

// ============================================================================
// Top-level implementation
// ============================================================================

void patch_ops_impl(
    fm_block_t dst[],
    const fm_block_t src[],
    PatchOpMode mode,
    unsigned int num_patches,
    unsigned int cls_token_idx,
    unsigned int inner_dim,
    unsigned int model_dim
)
{
    #pragma HLS inline off
    
    if (mode == PATCH_OP_FLIP) {
        patch_ops_flip_impl(dst, src, num_patches, inner_dim);
    } else {
        patch_ops_load_cls_impl(dst, src, num_patches, cls_token_idx, model_dim);
    }
}

#endif // __INT_PATCH_OPS_H__
