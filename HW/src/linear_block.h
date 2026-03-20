#ifndef __LINEAR_BLOCK_H__
#define __LINEAR_BLOCK_H__

#include "common.h"
#include "max.h"
#include "activation.h"
#include <hls_stream.h>
#include <cstring>

// ============================================================================
// Type Definitions
// ============================================================================

constexpr unsigned int WEIGHT_BLOCK_SIZE = 32;
constexpr unsigned int SCALES_PER_ROW = MAX_LINEAR_IN_DIM / WEIGHT_BLOCK_SIZE;
constexpr unsigned int MAX_SCALES_DIM = MAX_LINEAR_OUT_DIM * SCALES_PER_ROW;

// Weight packing constants
constexpr unsigned int WEIGHTS_PER_WORD = 64;  // 64 4-bit weights per 256-bit word

typedef ap_int<8> linear_quant_t;
typedef ap_int<16> linear_shift_t;
typedef ap_int<32> linear_sum_t;

typedef hls::vector<fm_t, LINEAR_BLOCK_SIZE> linear_in_t;
typedef hls::vector<fm_t, LINEAR_BLOCK_SIZE> linear_out_t;
typedef hls::vector<linear_quant_t, LINEAR_BLOCK_SIZE> linear_in_quant_t;
typedef hls::vector<linear_sum_t, LINEAR_BLOCK_SIZE> linear_out_quant_t;

// APoT LUT: 16 activations × 8 magnitude entries each (wrapped in struct for streaming)
typedef struct {
    linear_shift_t entries[LINEAR_BLOCK_SIZE][8];
} apot_lut_t;

// Weight block is 16×16 tile (reordered format)
typedef hls::vector<wt_linear_t, LINEAR_BLOCK_SIZE> wt_linear_weight_row_t;  // Row of 16 weights
typedef hls::vector<wt_linear_weight_row_t, LINEAR_BLOCK_SIZE> wt_linear_weight_block_t;  // 16×16 block

typedef hls::vector<wt_linear_bias_t, LINEAR_BLOCK_SIZE> wt_linear_bias_block_t;
typedef hls::vector<wt_linear_ws_t, LINEAR_BLOCK_SIZE> wt_linear_ws_block_t;
typedef hls::vector<wt_linear_ss_t, LINEAR_BLOCK_SIZE> wt_linear_ss_block_t;

// Flag constants (same as linear.h)
constexpr int FLAG_BIAS = 1;
constexpr int FLAG_SILU = 2;
constexpr int FLAG_SOFTPLUS = 4;

// ============================================================================
// Function Implementations
// ============================================================================

void load_linear_weights(
    wt_linear_weight_block_t dst[],
    const wt_wide_t src[],
    unsigned int out_dim,
    unsigned int in_dim
)
{
    #pragma HLS inline off
    
    // Weights are reordered into 16×16 blocks
    // Each 16×16 block has 256 4-bit weights = 1024 bits = 4 256-bit words
    unsigned int out_dim_blocks = ceildiv(out_dim, LINEAR_BLOCK_SIZE);
    unsigned int in_dim_blocks = ceildiv(in_dim, LINEAR_BLOCK_SIZE);
    unsigned int total_dim_blocks = out_dim_blocks * in_dim_blocks;
    
    FOR_EACH(total_block, total_dim_blocks)
    {
        #pragma HLS pipeline II=1
        
        // Each block needs 4 words (256 weights × 4 bits / 256 bits/word = 4)
        unsigned int word_base = total_block << 2;  // Multiply by 4
        
        wt_wide_t words[4];
        #pragma HLS array_partition variable=words complete
        FOR_EACH(w, 4)
        {
            #pragma HLS unroll
            words[w] = src[word_base + w];
        }
        
        // Unpack 256 weights into 16×16 block
        FOR_EACH(out_offset, LINEAR_BLOCK_SIZE)
        {
            #pragma HLS unroll
            FOR_EACH(in_offset, LINEAR_BLOCK_SIZE)
            {
                #pragma HLS unroll
                // Flat index within this 16×16 block
                unsigned int flat_idx = out_offset * LINEAR_BLOCK_SIZE + in_offset;
                
                // Determine which word and nibble position
                unsigned int word_idx = flat_idx >> 6;  // Divide by 64
                unsigned int nib_pos = flat_idx & 63;    // Mod 64
                unsigned int bit_start = nib_pos << 2;   // Multiply by 4
                
                wt_linear_t weight_val = (wt_linear_t)words[word_idx].range(bit_start + 3, bit_start);
                dst[total_block][out_offset][in_offset] = weight_val;
            }
        }
    }
}

constexpr unsigned int BIAS_PER_WORD = AXI_XFER_BIT_WIDTH / 32;  // 8 elements per 256-bit word

void load_linear_bias(
    wt_linear_bias_block_t dst[],
    const wt_wide_t src[],
    unsigned int out_dim
)
{
    #pragma HLS inline off
    
    static_assert(BIAS_PER_WORD * 2 == LINEAR_BLOCK_SIZE, 
                  "2 * BIAS_PER_WORD must equal LINEAR_BLOCK_SIZE for direct mapping");
    
    unsigned int num_blocks = ceildiv(out_dim, LINEAR_BLOCK_SIZE);
    
    FOR_EACH(block_idx, num_blocks)
    {
        #pragma HLS pipeline II=1
        unsigned int word_base = block_idx * 2;
        
        wt_wide_t word0 = src[word_base];
        wt_wide_t word1 = src[word_base + 1];
        
        FOR_EACH(j, BIAS_PER_WORD)
        {
            #pragma HLS unroll
            unsigned int idx = block_idx * LINEAR_BLOCK_SIZE + j;
            ap_uint<32> elem_bits = word0.range(j * 32 + 31, j * 32);
            dst[block_idx][j].range(31, 0) = elem_bits;
        }
        
        FOR_EACH(j, BIAS_PER_WORD)
        {
            #pragma HLS unroll
            unsigned int idx = block_idx * LINEAR_BLOCK_SIZE + BIAS_PER_WORD + j;
            ap_uint<32> elem_bits = word1.range(j * 32 + 31, j * 32);
            dst[block_idx][BIAS_PER_WORD + j].range(31, 0) = elem_bits;
        }
    }
}

constexpr unsigned int SCALES_PER_WORD = AXI_XFER_BIT_WIDTH / 32; // 8 elements per 256-bit word

void load_linear_weight_scales(
    wt_linear_ws_block_t dst[],
    const wt_wide_t src[],
    unsigned int out_dim,
    unsigned int in_dim
)
{
    #pragma HLS inline off

    static_assert(SCALES_PER_WORD * 2 == LINEAR_BLOCK_SIZE, 
                  "2 * SCALES_PER_WORD must equal LINEAR_BLOCK_SIZE for direct mapping");
    
    unsigned int scales_per_row = ceildiv(in_dim, WEIGHT_BLOCK_SIZE);
    unsigned int total_scales = scales_per_row * out_dim;
    unsigned int num_blocks = ceildiv(total_scales, LINEAR_BLOCK_SIZE);

    FOR_EACH(block_idx, num_blocks)
    {
        #pragma HLS pipeline II=1
        unsigned int word_base = block_idx * 2;
        
        wt_wide_t word0 = src[word_base];
        wt_wide_t word1 = src[word_base + 1];
        
        // Unpack first 8 scales from word0
        FOR_EACH(j, SCALES_PER_WORD)
        {
            #pragma HLS unroll
            ap_uint<32> elem_bits = word0.range(j * 32 + 31, j * 32);
            dst[block_idx][j].range(31, 0) = elem_bits;
        }
        
        // Unpack next 8 scales from word1
        FOR_EACH(j, SCALES_PER_WORD)
        {
            #pragma HLS unroll
            ap_uint<32> elem_bits = word1.range(j * 32 + 31, j * 32);
            dst[block_idx][SCALES_PER_WORD + j].range(31, 0) = elem_bits;
        }
    }
}

// ============================================================================
// Stage Functions (Streaming Architecture)
// ============================================================================

void read_in_stream(
    hls::stream<linear_in_t>& in_stream,
    hls::stream<wt_linear_as_t>& act_scales_quant_stream,
    hls::stream<wt_linear_as_t>& act_scales_dequant_stream,
    const fm_block_t src[],
    unsigned int in_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    static_assert(LINEAR_BLOCK_SIZE % FEATURE_BLOCK_SIZE == 0, "LINEAR_BLOCK_SIZE must be a multiple of FEATURE_BLOCK_SIZE");
    
    unsigned int in_dim_iters = ceildiv(in_dim, LINEAR_BLOCK_SIZE);
    unsigned int last_in_dim_iter = in_dim_iters - 1;
    constexpr unsigned int num_blocks = LINEAR_BLOCK_SIZE / FEATURE_BLOCK_SIZE;
    constexpr unsigned int last_block = num_blocks - 1;

    unsigned int next_in_dim_block = 0;
    unsigned int next_block = 0;
    unsigned int next_patch = 0;

    linear_in_t patch_buffer[ceildiv(MAX_LINEAR_IN_DIM, LINEAR_BLOCK_SIZE)];

    fm_block_t blocks[num_blocks];
    #pragma HLS array_partition variable=blocks complete

    fm_t tmp_max, patch_max;
    wt_linear_as_t tmp_act_scale;

    unsigned int iters = (num_patches + 1) * ceildiv(in_dim, FEATURE_BLOCK_SIZE);
    unsigned int write_iters = ceildiv(in_dim, FEATURE_BLOCK_SIZE);
    unsigned int read_iters = num_patches * ceildiv(in_dim, FEATURE_BLOCK_SIZE);

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1 style=flp
        #pragma HLS loop_tripcount min=1 max=((NUM_PATCHES+1)*ceildiv(D_INNER,FEATURE_BLOCK_SIZE))

        unsigned int block = next_block;
        next_block = (block == last_block) ? 0 : block + 1;

        unsigned int in_dim_block = next_in_dim_block;
        next_in_dim_block = (block == last_block)
            ? ((in_dim_block == last_in_dim_iter) ? 0 : in_dim_block + 1)
            : in_dim_block;

        unsigned int patch = next_patch;
        next_patch = (block == last_block)
            ? ((in_dim_block == last_in_dim_iter) ? patch + 1 : patch)
            : patch;

        if (i < read_iters)
        {
            blocks[block] = src[i];
        }

        if (block == last_block)
        {
            #pragma HLS occurrence cycle=num_blocks

            if (i >= write_iters)
            {
                in_stream << patch_buffer[in_dim_block];
            }

            if (i < read_iters)
            {
                linear_in_t stream_block;
                FOR_BLOCK(j, LINEAR_BLOCK_SIZE, FEATURE_BLOCK_SIZE)
                {
                    FOR_OFFSET_NOCHK(j)
                    {
                        #pragma HLS unroll
                        stream_block[j] = blocks[j_block][j_offset];
                    }
                }
                patch_buffer[in_dim_block] = stream_block;
                
                tmp_max = max_block(stream_block);
                patch_max = (in_dim_block == 0) ? tmp_max : (patch_max > tmp_max ? patch_max : tmp_max);
                tmp_act_scale = patch_max * (wt_linear_as_t)Q_MAX_FLOAT;
                if (in_dim_block == last_in_dim_iter) {
                    act_scales_quant_stream << tmp_act_scale;
                    act_scales_dequant_stream << tmp_act_scale;
                }
            }
        }
    }
}

void quantize_on_stream_linear(
    hls::stream<linear_in_quant_t>& in_quant_stream,
    hls::stream<linear_in_t>& in_stream,
    hls::stream<wt_linear_as_t>& act_scales_stream,
    unsigned int in_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    unsigned int in_dim_iters = ceildiv(in_dim, LINEAR_BLOCK_SIZE);
    unsigned int last_in_dim_iter = in_dim_iters - 1;
    unsigned int next_in_dim_block = 0;
    unsigned int next_patch = 0;

    linear_in_t block;
    linear_in_quant_t quantized_block;
    
    wt_linear_as_t current_act_scale;
    fm_t current_act_scale_reciprocal;

    unsigned int iters = num_patches * ceildiv(in_dim, LINEAR_BLOCK_SIZE);

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1 style=flp
        #pragma HLS loop_tripcount min=(1*ceildiv(DT_RANK,LINEAR_BLOCK_SIZE)) max=(NUM_PATCHES*ceildiv(D_INNER,LINEAR_BLOCK_SIZE))

        unsigned int in_dim_block = next_in_dim_block;
        next_in_dim_block = (in_dim_block == last_in_dim_iter) ? 0 : in_dim_block + 1;

        unsigned int patch = next_patch;
        next_patch = (in_dim_block == last_in_dim_iter) ? patch + 1 : patch;

        in_stream >> block;

        if (in_dim_block == 0)
        {
            act_scales_stream >> current_act_scale;
            current_act_scale_reciprocal = (fm_t)1.0 / (fm_t)current_act_scale;
        }

        scale_and_round: FOR_EACH(offset, LINEAR_BLOCK_SIZE)
        {
            #pragma HLS unroll factor=4
            fm_t scaled_value = block[offset] * current_act_scale_reciprocal;
            
            fm_t rounded_value;
            if (scaled_value >= 0) {
                rounded_value = scaled_value + (fm_t)0.5;
            } else {
                rounded_value = scaled_value - (fm_t)0.5;
            }
            
            linear_quant_t quantized_value;
            if (rounded_value >= (fm_t)127.5) {
                quantized_value = (linear_quant_t)127;
            } else if (rounded_value <= (fm_t)-128.5) {
                quantized_value = (linear_quant_t)-128;
            } else {
                quantized_value = (linear_quant_t)rounded_value.to_int();
            }
            
            quantized_block[offset] = quantized_value;
        }
        in_quant_stream << quantized_block;
    }
}

typedef struct {
    linear_sum_t sums[LINEAR_BLOCK_SIZE];
    unsigned int out_dim_block;
    unsigned int block_32_idx;
    bool is_last_block;
} partial_sum_block_t;

typedef struct {
    unsigned int in_dim_block_16;
    unsigned int out_dim_block;
    bool is_last_tile;
} tile_info_t;

void decode_weights_on_stream(
    hls::stream<tile_info_t>& tile_info_stream,
    hls::stream<apot_lut_t>& apot_lut_stream,
    hls::stream<linear_in_quant_t>& in_quant_stream,
    const wt_linear_weight_block_t weights[],
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int num_patches
) {
    #pragma HLS inline off
    
    const unsigned int out_dim_iters = ceildiv(out_dim, LINEAR_BLOCK_SIZE);
    const unsigned int last_out_dim_iter = out_dim_iters - 1;
    const unsigned int in_dim_iters_16 = ceildiv(in_dim, LINEAR_BLOCK_SIZE);
    const unsigned int last_in_dim_iter_16 = in_dim_iters_16 - 1;
    const unsigned int total_dim_iters = out_dim_iters * in_dim_iters_16;
    const unsigned int last_total_dim_iter = total_dim_iters - 1;
    
    unsigned int iters = num_patches * total_dim_iters;
    
    unsigned int next_total_dim_block = 0;
    unsigned int next_in_dim_block_16 = 0;
    unsigned int next_out_dim_block = 0;

    linear_in_quant_t in_blocks[ceildiv(MAX_LINEAR_IN_DIM, LINEAR_BLOCK_SIZE)];
    #pragma HLS array_partition variable=in_blocks cyclic factor=4
    
    FOR_EACH(i, iters) {
        #pragma HLS pipeline II=1 style=flp
        #pragma HLS loop_tripcount min=(1*ceildiv(DT_RANK,LINEAR_BLOCK_SIZE)*ceildiv(DT_RANK,LINEAR_BLOCK_SIZE)) max=(NUM_PATCHES*ceildiv(PADDED_NUM_CLASSES,LINEAR_BLOCK_SIZE)*ceildiv(D_INNER,LINEAR_BLOCK_SIZE))
        
        unsigned int total_dim_block = next_total_dim_block;
        next_total_dim_block = (total_dim_block == last_total_dim_iter) ? 0 : total_dim_block + 1;
        
        unsigned int in_dim_block_16 = next_in_dim_block_16;
        next_in_dim_block_16 = (in_dim_block_16 == last_in_dim_iter_16) ? 0 : in_dim_block_16 + 1;
        
        unsigned int out_dim_block = next_out_dim_block;
        next_out_dim_block = (total_dim_block == last_total_dim_iter) ? 0 :
                             (in_dim_block_16 == last_in_dim_iter_16) ? out_dim_block + 1 : out_dim_block;
        
        // Read input tile when starting new output row
        if (out_dim_block == 0) {
            in_quant_stream >> in_blocks[in_dim_block_16];
        }
        linear_in_quant_t in_tile = in_blocks[in_dim_block_16];
        
        // Build APoT LUT for this input tile
        apot_lut_t lut;
        #pragma HLS array_partition variable=lut.entries complete dim=2
        
        FOR_EACH(ic, LINEAR_BLOCK_SIZE) {
            #pragma HLS unroll
            linear_shift_t x = (linear_shift_t)in_tile[ic];
            linear_shift_t s4 = x << 4;
            linear_shift_t s5 = x << 5;
            linear_shift_t s6 = x << 6;
            linear_shift_t s7 = x << 7;
            
            lut.entries[ic][0] = 0;
            lut.entries[ic][1] = s7;
            lut.entries[ic][2] = s6;
            lut.entries[ic][3] = s4;
            lut.entries[ic][4] = s5;
            lut.entries[ic][5] = s7 + s5;
            lut.entries[ic][6] = s6 + s5;
            lut.entries[ic][7] = s4 + s5;
        }
        
        apot_lut_stream << lut;
        
        tile_info_t tile_info;
        tile_info.in_dim_block_16 = in_dim_block_16;
        tile_info.out_dim_block = out_dim_block;
        tile_info.is_last_tile = (in_dim_block_16 == last_in_dim_iter_16);
        
        tile_info_stream << tile_info;
    }
}

void compute_mac_on_stream(
    hls::stream<partial_sum_block_t>& partial_sum_stream,
    hls::stream<tile_info_t>& tile_info_stream,
    hls::stream<apot_lut_t>& apot_lut_stream,
    const wt_linear_weight_block_t weights[],
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int num_patches
) {
    #pragma HLS inline off
    
    const unsigned int out_dim_iters = ceildiv(out_dim, LINEAR_BLOCK_SIZE);
    const unsigned int in_dim_iters_16 = ceildiv(in_dim, LINEAR_BLOCK_SIZE);
    const unsigned int total_dim_iters = out_dim_iters * in_dim_iters_16;
    
    unsigned int iters = num_patches * total_dim_iters;

    linear_sum_t partial_sums[LINEAR_BLOCK_SIZE];
    #pragma HLS array_partition variable=partial_sums complete
    
    FOR_EACH(i, iters) {
        #pragma HLS pipeline II=1 style=flp
        #pragma HLS loop_tripcount min=(1*ceildiv(DT_RANK,LINEAR_BLOCK_SIZE)*ceildiv(DT_RANK,LINEAR_BLOCK_SIZE)) max=(NUM_PATCHES*ceildiv(PADDED_NUM_CLASSES,LINEAR_BLOCK_SIZE)*ceildiv(D_INNER,LINEAR_BLOCK_SIZE))
        
        tile_info_t tile_info;
        tile_info_stream >> tile_info;
        
        apot_lut_t lut;
        #pragma HLS array_partition variable=lut.entries complete dim=2
        apot_lut_stream >> lut;
        
        bool is_even_tile = (tile_info.in_dim_block_16 & 1) == 0;
        
        // At start of each 32-element block, clear partial sums
        if (is_even_tile || tile_info.in_dim_block_16 == 0) {
            FOR_EACH(oc, LINEAR_BLOCK_SIZE) {
                #pragma HLS unroll
                partial_sums[oc] = 0;
            }
        }
        
        // Load weight block
        unsigned int weight_block_idx = tile_info.out_dim_block * in_dim_iters_16 + tile_info.in_dim_block_16;
        wt_linear_weight_block_t weight_block = weights[weight_block_idx];
        
        // Compute for all 16 output channels in parallel
        FOR_EACH(oc, LINEAR_BLOCK_SIZE) {
            #pragma HLS unroll
            
            // MAC over 16 input elements with LUT-based APoT decode
            linear_sum_t products[LINEAR_BLOCK_SIZE];
            #pragma HLS array_partition variable=products complete
            
            FOR_EACH(ic, LINEAR_BLOCK_SIZE) {
                #pragma HLS unroll
                
                wt_linear_t weight = weight_block[oc][ic];
                
                // Simple LUT-based APoT decode
                ap_uint<1> sign = (weight >> 3) & 0x1;
                ap_uint<3> mag_idx = weight & 0x7;
                
                linear_shift_t mag = lut.entries[ic][mag_idx];
                linear_shift_t signed_mag = sign ? (linear_shift_t)(-mag) : mag;
                
                products[ic] = (linear_sum_t)signed_mag;
            }
            
            // Adder tree: 16->8->4->2->1 (log depth, better than sequential)
            linear_sum_t sum_8[8], sum_4[4], sum_2[2];
            #pragma HLS array_partition variable=sum_8 complete
            #pragma HLS array_partition variable=sum_4 complete
            #pragma HLS array_partition variable=sum_2 complete
            
            FOR_EACH(k, 8) {
                #pragma HLS unroll
                sum_8[k] = products[k*2] + products[k*2+1];
            }
            FOR_EACH(k, 4) {
                #pragma HLS unroll
                sum_4[k] = sum_8[k*2] + sum_8[k*2+1];
            }
            FOR_EACH(k, 2) {
                #pragma HLS unroll
                sum_2[k] = sum_4[k*2] + sum_4[k*2+1];
            }
            
            partial_sums[oc] += sum_2[0] + sum_2[1];
        }
        
        // Every 2 tiles (32 elements), send partial sums to scaling stage
        bool is_odd_tile = (tile_info.in_dim_block_16 & 1) == 1;
        if (is_odd_tile || tile_info.is_last_tile) {
            partial_sum_block_t ps_block;
            FOR_EACH(oc, LINEAR_BLOCK_SIZE) {
                #pragma HLS unroll
                ps_block.sums[oc] = partial_sums[oc];
            }
            ps_block.out_dim_block = tile_info.out_dim_block;
            ps_block.block_32_idx = tile_info.in_dim_block_16 >> 1;
            ps_block.is_last_block = tile_info.is_last_tile;
            
            partial_sum_stream << ps_block;
        }
    }
}

void scale_and_accumulate_on_stream(
    hls::stream<linear_out_t>& out_quant_stream,
    hls::stream<partial_sum_block_t>& partial_sum_stream,
    const wt_linear_ws_block_t weight_scales[],
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int num_patches
) {
    #pragma HLS inline off
    
    const unsigned int out_dim_iters = ceildiv(out_dim, LINEAR_BLOCK_SIZE);
    const unsigned int in_dim_iters_32 = ceildiv(in_dim, WEIGHT_BLOCK_SIZE);
    unsigned int iters = num_patches * out_dim_iters * in_dim_iters_32;

    unsigned int ws_idx = 0;

    linear_out_t out_block;
    #pragma HLS array_partition variable=out_block complete
    
    FOR_EACH(i, iters) {
        #pragma HLS pipeline II=1 style=flp
        #pragma HLS loop_tripcount min=(1*ceildiv(DT_RANK,LINEAR_BLOCK_SIZE)*ceildiv(DT_RANK,WEIGHT_BLOCK_SIZE)) max=(NUM_PATCHES*ceildiv(PADDED_NUM_CLASSES,LINEAR_BLOCK_SIZE)*ceildiv(D_INNER,WEIGHT_BLOCK_SIZE))
        
        partial_sum_block_t ps_block;
        partial_sum_stream >> ps_block;

        if (ps_block.block_32_idx == 0) {
            FOR_EACH(oc, LINEAR_BLOCK_SIZE) {
                #pragma HLS unroll
                out_block[oc] = 0;
            }
        }

        ws_idx = (ps_block.block_32_idx == 0) ? ps_block.out_dim_block : (ws_idx + out_dim_iters);
        wt_linear_ws_block_t ws_block = weight_scales[ws_idx];

        FOR_EACH(oc, LINEAR_BLOCK_SIZE) {
            #pragma HLS unroll
            
            ap_fixed<64, 38> scaled = ps_block.sums[oc] * ws_block[oc];
            out_block[oc] += (fm_t)(scaled >> 8);
        }
        
        if (ps_block.is_last_block) {
            out_quant_stream << out_block;
        }
    }
}

void dequantize_on_stream_linear(
    hls::stream<linear_out_t>& out_stream,
    hls::stream<linear_out_t>& out_quant_stream,
    hls::stream<wt_linear_as_t>& act_scales_stream,
    const wt_linear_bias_block_t bias[],
    unsigned int out_dim,
    unsigned int num_patches,
    int flags
)
{
    #pragma HLS inline off

    bool if_bias = flags & FLAG_BIAS;
    bool if_silu = flags & FLAG_SILU;
    bool if_softplus = flags & FLAG_SOFTPLUS;

    unsigned int out_dim_iters = ceildiv(out_dim, LINEAR_BLOCK_SIZE);
    unsigned int last_out_dim_iter = out_dim_iters - 1;
    unsigned int next_out_dim_block = 0;
    unsigned int next_patch = 0;

    wt_linear_as_t current_act_scale;

    unsigned int iters = num_patches * out_dim_iters;

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1 style=flp
        #pragma HLS loop_tripcount min=(1*ceildiv(DT_RANK,LINEAR_BLOCK_SIZE)) max=(NUM_PATCHES*ceildiv(PADDED_NUM_CLASSES,LINEAR_BLOCK_SIZE))

        unsigned int out_dim_block = next_out_dim_block;
        next_out_dim_block = (out_dim_block == last_out_dim_iter) ? 0 : out_dim_block + 1;

        unsigned int patch = next_patch;
        next_patch = (out_dim_block == last_out_dim_iter) ? patch + 1 : patch;

        if (out_dim_block == 0)
        {
            act_scales_stream >> current_act_scale;
        }

        linear_out_t out_block;
        out_quant_stream >> out_block;

        linear_out_t dequant_block;
        
        FOR_EACH(dim_offset, LINEAR_BLOCK_SIZE)
        {
            #pragma HLS unroll factor=4
            #pragma HLS dependence variable=bias inter false
            
            fm_t final_result = out_block[dim_offset] * current_act_scale;
            final_result = (if_bias) ? (fm_t)(final_result + bias[out_dim_block][dim_offset]) : final_result;
            dequant_block[dim_offset] = final_result;
        }

        dequant_block = (if_silu) ? silu(dequant_block) : dequant_block;
        dequant_block = (if_softplus) ? softplus(dequant_block) : dequant_block;

        out_stream << dequant_block;
    }
}

void write_out_stream_linear(
    fm_block_t dst[],
    hls::stream<linear_out_t>& out_stream,
    unsigned int out_dim,
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

    unsigned int iters = num_patches * ceildiv(out_dim, FEATURE_BLOCK_SIZE);

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1 style=flp
        #pragma HLS loop_tripcount min=(1*ceildiv(DT_RANK,FEATURE_BLOCK_SIZE)) max=(NUM_PATCHES*ceildiv(PADDED_NUM_CLASSES,FEATURE_BLOCK_SIZE))

        unsigned int block = next_block;
        next_block = (block == last_block) ? 0 : block + 1;

        if (block == 0)
        {
            #pragma HLS occurrence cycle=num_blocks

            linear_out_t stream_block;
            out_stream >> stream_block;
            FOR_BLOCK(j, LINEAR_BLOCK_SIZE, FEATURE_BLOCK_SIZE)
            {
                fm_block_t slice;
                FOR_OFFSET_NOCHK(j)
                {
                    #pragma HLS unroll factor=4
                    slice[j_offset] = stream_block[j];
                }
                blocks[j_block] = slice;
            }
        }

        dst[i] = blocks[block];
    }
}

// ============================================================================
// Top-Level Function (with wt_wide_t interface like linear.h)
// ============================================================================

void compute_linear_block_impl(
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_linear_weight_block_t weights[],
    const wt_linear_ws_block_t weight_scales[],
    const wt_linear_bias_block_t bias[],
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int num_patches,
    int flags
)
{
    #pragma HLS inline off
    #pragma HLS dataflow

    hls::stream<linear_in_t> in_stream("linear_in_stream");
    #pragma HLS stream variable=in_stream depth=(MAX_LINEAR_IN_DIM / LINEAR_BLOCK_SIZE)
    hls::stream<linear_in_quant_t> in_quant_stream("linear_in_quant_stream");
    #pragma HLS stream variable=in_quant_stream depth=2
    hls::stream<apot_lut_t> apot_lut_stream("apot_lut_stream");
    #pragma HLS stream variable=apot_lut_stream depth=2
    hls::stream<tile_info_t> tile_info_stream("tile_info_stream");
    #pragma HLS stream variable=tile_info_stream depth=2
    hls::stream<partial_sum_block_t> partial_sum_stream("partial_sum_stream");
    #pragma HLS stream variable=partial_sum_stream depth=2
    hls::stream<linear_out_t> out_quant_stream("linear_out_quant_stream");
    #pragma HLS stream variable=out_quant_stream depth=2
    hls::stream<linear_out_t> out_stream("linear_out_stream");
    #pragma HLS stream variable=out_stream depth=2
    hls::stream<wt_linear_as_t> act_scales_quant_stream("act_scales_quant_stream");
    #pragma HLS stream variable=act_scales_quant_stream depth=2
    hls::stream<wt_linear_as_t> act_scales_dequant_stream("act_scales_dequant_stream");
    #pragma HLS stream variable=act_scales_dequant_stream depth=2

    read_in_stream(in_stream, act_scales_quant_stream, act_scales_dequant_stream, src, in_dim, num_patches);
    quantize_on_stream_linear(in_quant_stream, in_stream, act_scales_quant_stream, in_dim, num_patches);
    decode_weights_on_stream(tile_info_stream, apot_lut_stream, in_quant_stream, weights, out_dim, in_dim, num_patches);
    compute_mac_on_stream(partial_sum_stream, tile_info_stream, apot_lut_stream, weights, out_dim, in_dim, num_patches);
    scale_and_accumulate_on_stream(out_quant_stream, partial_sum_stream, weight_scales, out_dim, in_dim, num_patches);
    dequantize_on_stream_linear(out_stream, out_quant_stream, act_scales_dequant_stream, bias, out_dim, num_patches, flags);
    write_out_stream_linear(dst, out_stream, out_dim, num_patches);
}

void compute_linear_block(
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_wide_t weights_base[],
    const wt_wide_t weight_scales_base[],
    const wt_wide_t bias_base[],
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int num_patches,
    int flags
) {
    #pragma HLS inline off

    static wt_linear_weight_block_t weights[ceildiv(MAX_LINEAR_OUT_DIM, LINEAR_BLOCK_SIZE) * ceildiv(MAX_LINEAR_IN_DIM, LINEAR_BLOCK_SIZE)]; 
    static wt_linear_bias_block_t bias[ceildiv(MAX_LINEAR_OUT_DIM, LINEAR_BLOCK_SIZE)];
    static wt_linear_ws_block_t weight_scales[ceildiv(MAX_SCALES_DIM, LINEAR_BLOCK_SIZE)];
    #pragma HLS array_partition variable=weight_scales cyclic factor=16
    
    load_linear_weights(weights, weights_base, out_dim, in_dim);
    load_linear_weight_scales(weight_scales, weight_scales_base, out_dim, in_dim);
    if (flags & FLAG_BIAS) {
        load_linear_bias(bias, bias_base, out_dim);
    }
    
    compute_linear_block_impl(dst, src, weights, weight_scales, bias, out_dim, in_dim, num_patches, flags);
}

#endif // __LINEAR_BLOCK_H__
