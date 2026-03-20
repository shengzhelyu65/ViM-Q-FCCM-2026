#ifndef __INT_CONV_H__
#define __INT_CONV_H__

#include "common.h"
#include "max.h"
#include "activation.h"
#include <hls_stream.h>

// ============================================================================
// Type Definitions
// ============================================================================

typedef ap_int<8> conv_quant_t;
typedef ap_int<16> conv_shift_t;
typedef ap_int<32> conv_sum_t;

typedef hls::vector<fm_t, CONV_BLOCK_SIZE> conv_in_t;
typedef hls::vector<fm_t, CONV_BLOCK_SIZE> conv_out_t;
typedef hls::vector<conv_quant_t, CONV_BLOCK_SIZE> conv_in_quant_block_t;
typedef hls::vector<conv_in_quant_block_t, CONV_KERNEL_SIZE> conv_in_quant_t;
typedef hls::vector<conv_sum_t, CONV_BLOCK_SIZE> conv_out_quant_t;

typedef hls::vector<wt_conv_bias_t, CONV_BLOCK_SIZE> wt_conv_bias_block_t;
typedef hls::vector<wt_conv_ws_t, CONV_BLOCK_SIZE> wt_conv_ws_block_t;

// Pre-unpacked weight blocks for efficient computation
// Format: [dim_block][dim_offset][kernel_offset]
typedef hls::vector<wt_conv_sign_t, CONV_KERNEL_SIZE> wt_conv_sign_kernel_t;
typedef hls::vector<wt_conv_mag_t, CONV_KERNEL_SIZE> wt_conv_mag_kernel_t;
typedef hls::vector<wt_conv_sign_kernel_t, CONV_BLOCK_SIZE> wt_conv_weight_sign_block_t;  // [dim_offset][kernel_offset]
typedef hls::vector<wt_conv_mag_kernel_t, CONV_BLOCK_SIZE> wt_conv_weight_mag_block_t;    // [dim_offset][kernel_offset]

// ============================================================================
// Function Implementations
// ============================================================================

// Load weights directly from base arrays into block format
void load_conv_weights_magnitudes(
    wt_conv_weight_mag_block_t dst[],
    const wt_wide_t src[],
    unsigned int conv_dim
)
{
    #pragma HLS inline off
    
    unsigned int dim_blocks = ceildiv(conv_dim, CONV_BLOCK_SIZE);

    FOR_EACH(dim_block, dim_blocks)
    {
        #pragma HLS pipeline II=1
        
        FOR_EACH(dim_offset, CONV_BLOCK_SIZE)
        {
            #pragma HLS unroll
            unsigned int dim_idx = dim_block * CONV_BLOCK_SIZE + dim_offset;
            
            FOR_EACH(kernel_offset, CONV_KERNEL_SIZE)
            {
                #pragma HLS unroll
                unsigned int flat_idx = dim_idx * CONV_KERNEL_SIZE + kernel_offset;
                
                unsigned int mag_word_idx = flat_idx >> 6;
                unsigned int mag_bit_offset = (flat_idx & 63) << 2;
                
                wt_wide_t mag_word = src[mag_word_idx];
                wt_conv_mag_t mag_val = (wt_conv_mag_t)mag_word.range(mag_bit_offset + 3, mag_bit_offset);
                dst[dim_block][dim_offset][kernel_offset] = mag_val;
            }
        }
    }
}

void load_conv_weights_signs(
    wt_conv_weight_sign_block_t dst[],
    const wt_wide_t src[],
    unsigned int conv_dim
)
{
    #pragma HLS inline off
    
    unsigned int dim_blocks = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    
    FOR_EACH(dim_block, dim_blocks)
    {
        #pragma HLS pipeline II=1
        
        FOR_EACH(dim_offset, CONV_BLOCK_SIZE)
        {
            #pragma HLS unroll
            unsigned int dim_idx = dim_block * CONV_BLOCK_SIZE + dim_offset;
            
            FOR_EACH(kernel_offset, CONV_KERNEL_SIZE)
            {
                #pragma HLS unroll
                unsigned int flat_idx = dim_idx * CONV_KERNEL_SIZE + kernel_offset;
                
                unsigned int sign_word_idx = flat_idx >> 8;
                unsigned int sign_bit_offset = flat_idx & 255;
                
                wt_wide_t sign_word = src[sign_word_idx];
                wt_conv_sign_t sign_val = (wt_conv_sign_t)sign_word[sign_bit_offset];
                dst[dim_block][dim_offset][kernel_offset] = sign_val;
            }
        }
    }
}

constexpr unsigned int CONV_BIAS_PER_WORD = AXI_XFER_BIT_WIDTH / 32;  // 8 elements per 256-bit word

void load_conv_bias(
    wt_conv_bias_block_t dst[],
    const wt_wide_t src[],
    unsigned int conv_dim
)
{
    #pragma HLS inline off

    static_assert(CONV_BIAS_PER_WORD * 2 == CONV_BLOCK_SIZE, 
                  "2 * CONV_BIAS_PER_WORD must equal CONV_BLOCK_SIZE for direct mapping");
    
    unsigned int num_blocks = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    unsigned int num_word_pairs = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    
    FOR_EACH(block_idx, num_word_pairs)
    {
        #pragma HLS pipeline II=1
        unsigned int word_base = block_idx * 2;
        
        wt_wide_t word0 = src[word_base];
        wt_wide_t word1 = src[word_base + 1];
        
        FOR_EACH(j, CONV_BIAS_PER_WORD)
        {
            #pragma HLS unroll
            unsigned int idx = block_idx * CONV_BLOCK_SIZE + j;
            ap_uint<32> elem_bits = word0.range(j * 32 + 31, j * 32);
            dst[block_idx][j].range(31, 0) = elem_bits;
        }
        
        // Unpack second word (8 elements) into second half of block
        FOR_EACH(j, CONV_BIAS_PER_WORD)
        {
            #pragma HLS unroll
            unsigned int idx = block_idx * CONV_BLOCK_SIZE + CONV_BIAS_PER_WORD + j;
            ap_uint<32> elem_bits = word1.range(j * 32 + 31, j * 32);
            dst[block_idx][CONV_BIAS_PER_WORD + j].range(31, 0) = elem_bits;
        }
    }
}

constexpr unsigned int CONV_SCALES_PER_WORD = AXI_XFER_BIT_WIDTH / 32; // 8 elements per 256-bit word

void load_conv_weight_scales(
    wt_conv_ws_block_t dst[],
    const wt_wide_t src[],
    unsigned int conv_dim
)
{
    #pragma HLS inline off
    
    static_assert(CONV_SCALES_PER_WORD * 2 == CONV_BLOCK_SIZE, 
                  "2 * CONV_SCALES_PER_WORD must equal CONV_BLOCK_SIZE for direct mapping");
    
    unsigned int num_blocks = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    unsigned int num_word_pairs = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    
    FOR_EACH(block_idx, num_word_pairs)
    {
        #pragma HLS pipeline II=1
        unsigned int word_base = block_idx * 2;
        
        wt_wide_t word0 = src[word_base];
        wt_wide_t word1 = src[word_base + 1];
        
        FOR_EACH(j, CONV_SCALES_PER_WORD)
        {
            #pragma HLS unroll
            unsigned int idx = block_idx * CONV_BLOCK_SIZE + j;
            ap_uint<32> elem_bits = word0.range(j * 32 + 31, j * 32);
            dst[block_idx][j].range(31, 0) = elem_bits;
        }
        
        // Unpack second word (8 elements) into second half of block
        FOR_EACH(j, CONV_SCALES_PER_WORD)
        {
            #pragma HLS unroll
            unsigned int idx = block_idx * CONV_BLOCK_SIZE + CONV_SCALES_PER_WORD + j;
            ap_uint<32> elem_bits = word1.range(j * 32 + 31, j * 32);
            dst[block_idx][CONV_SCALES_PER_WORD + j].range(31, 0) = elem_bits;
        }
    }
}
void read_in_stream(
    hls::stream<conv_in_t>& in_stream,
    hls::stream<wt_conv_as_t>& act_scale_stream1,
    hls::stream<wt_conv_as_t>& act_scale_stream2,
    const fm_block_t src[],
    unsigned int conv_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    static_assert(CONV_BLOCK_SIZE % FEATURE_BLOCK_SIZE == 0, "CONV_BLOCK_SIZE must be a multiple of FEATURE_BLOCK_SIZE");

    unsigned int in_dim_iters = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    unsigned int last_in_dim_iter = in_dim_iters - 1;
    constexpr unsigned int num_blocks = CONV_BLOCK_SIZE / FEATURE_BLOCK_SIZE;
    constexpr unsigned int last_block = num_blocks - 1;
    
    unsigned int next_dim_block = 0;
    unsigned int next_block = 0;

    unsigned int iters = num_patches * ceildiv(conv_dim, FEATURE_BLOCK_SIZE);

    fm_block_t blocks[CONV_BLOCK_SIZE / FEATURE_BLOCK_SIZE];
    #pragma HLS array_partition variable=blocks cyclic factor=2

    fm_t max_regs[FEATURE_BLOCK_SIZE];
    #pragma HLS array_partition variable=max_regs complete
    FOR_EACH(j, FEATURE_BLOCK_SIZE) max_regs[j] = (fm_t)0;

    // Pass 1: Compute max for quantization scale
    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1
        fm_block_t current_block = src[i];
        FOR_EACH(j, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            fm_t val = current_block[j];
            fm_t abs_val = (val < (fm_t)0) ? (fm_t)-val : val;
            if (abs_val > max_regs[j]) max_regs[j] = abs_val;
        }
    }
    
    fm_t tensor_max = (fm_t)0;
    FOR_EACH(j, FEATURE_BLOCK_SIZE)
    {
        #pragma HLS unroll
        if (max_regs[j] > tensor_max) tensor_max = max_regs[j];
    }
    
    wt_conv_as_t act_scale = tensor_max * (wt_conv_as_t)Q_MAX_FLOAT;
    act_scale_stream1 << act_scale;
    act_scale_stream2 << act_scale;
    
    // Reset counters for Pass 2
    next_block = 0;
    next_dim_block = 0; // Though not used in loop 1 logic, good to be clean
    
    // Pass 2: Stream data
    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1

        unsigned int block = next_block;
        next_block = (block == last_block) ? 0 : block + 1;

        unsigned int dim_block = next_dim_block;
        next_dim_block = (block == last_block)
            ? ((dim_block == last_in_dim_iter) ? 0 : dim_block + 1)
            : dim_block;

        blocks[block] = src[i];

        if (block == last_block)
        {
            #pragma HLS occurrence cycle=num_blocks

            conv_in_t stream_conv_block;
            FOR_BLOCK(j, CONV_BLOCK_SIZE, FEATURE_BLOCK_SIZE)
            {
                FOR_OFFSET_NOCHK(j)
                {
                    #pragma HLS unroll factor=4
                    stream_conv_block[j] = blocks[j_block][j_offset];
                }
            }
            in_stream << stream_conv_block;
        }
    }
}

void window_and_quantize_on_stream(
    hls::stream<conv_in_quant_t>& in_quant_stream,
    hls::stream<conv_in_t>& in_stream,
    hls::stream<wt_conv_as_t>& act_scale_stream,
    unsigned int conv_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    wt_conv_as_t act_scale = act_scale_stream.read();

    conv_in_quant_block_t line_buffer[CONV_KERNEL_SIZE-1][ceildiv(D_INNER, CONV_BLOCK_SIZE)];
    #pragma HLS array_partition variable=line_buffer cyclic factor=4 dim=2
    
    conv_in_quant_t window;
    #pragma HLS array_partition variable=window cyclic factor=2 dim=1

    unsigned int dim_blocks = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    unsigned int last_dim_block = dim_blocks - 1;
    unsigned int next_dim_block = 0;

    unsigned int iters = num_patches * dim_blocks;

    FOR_EACH(k, CONV_KERNEL_SIZE-1)
    {
        FOR_EACH(j_block, dim_blocks)
        {
            FOR_EACH(j_offset, CONV_BLOCK_SIZE)
            {
                #pragma HLS unroll
                line_buffer[k][j_block][j_offset] = (conv_quant_t)0;
            }
        }
    }

    fm_t act_scale_reciprocal = (fm_t)1.0 / (fm_t)act_scale;

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1

        unsigned int dim_block = next_dim_block;
        next_dim_block = (dim_block == last_dim_block) ? 0 : dim_block + 1;

        conv_in_t current_block;
        in_stream >> current_block;

        conv_in_quant_block_t current_quant_block;
        FOR_EACH(k, CONV_BLOCK_SIZE)
        {
            #pragma HLS unroll factor=4
            fm_t scaled_value = current_block[k] * act_scale_reciprocal;
            
            fm_t rounded_value = scaled_value + (scaled_value >= (fm_t)0 ? (fm_t)0.5 : (fm_t)-0.5);
            
            conv_quant_t quantized_value;
            if (rounded_value >= (fm_t)127.5) {
                quantized_value = (conv_quant_t)127;
            } else if (rounded_value <= (fm_t)-128.5) {
                quantized_value = (conv_quant_t)-128;
            } else {
                quantized_value = (conv_quant_t)rounded_value.to_int();
            }
            
            current_quant_block[k] = quantized_value;
        }

        FOR_EACH(k, CONV_KERNEL_SIZE-1)
        {
            #pragma HLS unroll factor=4
            window[k] = line_buffer[k][dim_block];
        }
        window[CONV_KERNEL_SIZE-1] = current_quant_block;

        FOR_EACH(k, CONV_KERNEL_SIZE-2)
        {
            #pragma HLS unroll factor=4
            line_buffer[k][dim_block] = line_buffer[k+1][dim_block];
        }
        line_buffer[CONV_KERNEL_SIZE-2][dim_block] = current_quant_block;

        in_quant_stream << window;
    }
}

void compute_conv_on_stream(
    hls::stream<conv_out_quant_t>& out_stream,
    hls::stream<conv_in_quant_t>& in_stream,
    const wt_conv_weight_mag_block_t weights_mags[],
    const wt_conv_weight_sign_block_t weights_signs[],
    unsigned int conv_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    conv_in_quant_t in_window;
    conv_out_quant_t out_block;
    #pragma HLS array_partition variable=out_block cyclic factor=2

    unsigned int dim_blocks = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    unsigned int last_dim_block = dim_blocks - 1;
    unsigned int next_dim_block = 0;
    unsigned int iters = num_patches * dim_blocks;

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1

        unsigned int dim_block = next_dim_block;
        next_dim_block = (dim_block == last_dim_block) ? 0 : dim_block + 1;

        in_stream >> in_window;

        const wt_conv_weight_sign_block_t& sign_block = weights_signs[dim_block];
        const wt_conv_weight_mag_block_t& mag_block = weights_mags[dim_block];

        FOR_EACH(dim_offset, CONV_BLOCK_SIZE)
        {
            #pragma HLS unroll factor=4
            conv_sum_t out_block_temp = (conv_sum_t)0;

            FOR_EACH(kernel_offset, CONV_KERNEL_SIZE)
            {
                #pragma HLS unroll
                
                wt_conv_sign_t sign_val = sign_block[dim_offset][kernel_offset];
                wt_conv_mag_t mag_val = mag_block[dim_offset][kernel_offset];

                ap_uint<2> lower_bits = mag_val.range(1, 0);
                ap_uint<2> upper_bits = mag_val.range(3, 2);
                
                conv_quant_t in_val = in_window[kernel_offset][dim_offset];
                conv_shift_t shifted_a, shifted_b;

                switch(lower_bits) {
                    case 0: shifted_a = (conv_shift_t(in_val) >> 8); break;
                    case 1: shifted_a = (conv_shift_t(in_val) << 7); break;
                    case 2: shifted_a = (conv_shift_t(in_val) << 5); break;
                    case 3: shifted_a = (conv_shift_t(in_val) << 3); break;
                }

                switch(upper_bits) {
                    case 0: shifted_b = (conv_shift_t(in_val) >> 8); break;
                    case 1: shifted_b = (conv_shift_t(in_val) << 6); break;
                    case 2: shifted_b = (conv_shift_t(in_val) << 4); break;
                    case 3: shifted_b = (conv_shift_t(in_val) << 2); break;
                }

                conv_shift_t accumulator = shifted_a + shifted_b;
                conv_shift_t signed_result = (sign_val == (wt_conv_sign_t)0) ? accumulator : (conv_shift_t)(-accumulator);

                out_block_temp += (conv_sum_t)signed_result;
            }

            out_block[dim_offset] = out_block_temp >> 8;
        }

        out_stream << out_block;
    }
}

void dequantize_on_stream(
    hls::stream<conv_out_t>& out_stream,
    hls::stream<conv_out_quant_t>& out_quant_stream,
    const wt_conv_bias_block_t bias[],
    const wt_conv_ws_block_t weight_scales[],
    hls::stream<wt_conv_as_t>& act_scale_stream,
    unsigned int conv_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    
    wt_conv_as_t act_scale = act_scale_stream.read();

    conv_out_quant_t out_quant_block;
    conv_out_t out_block;
    #pragma HLS array_partition variable=out_block cyclic factor=4

    unsigned int dim_blocks = ceildiv(conv_dim, CONV_BLOCK_SIZE);
    unsigned int last_dim_block = dim_blocks - 1;
    unsigned int next_dim_block = 0;
    unsigned int iters = num_patches * dim_blocks;

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1

        unsigned int dim_block = next_dim_block;
        next_dim_block = (dim_block == last_dim_block) ? 0 : dim_block + 1;

        out_quant_stream >> out_quant_block;

        FOR_EACH(dim_offset, CONV_BLOCK_SIZE)
        {
            #pragma HLS unroll factor=4
            
            conv_sum_t quant_val = out_quant_block[dim_offset];
            wt_conv_ws_t weight_scale = weight_scales[dim_block][dim_offset];
            wt_conv_bias_t bias_val = bias[dim_block][dim_offset];
            
            ap_fixed<64, 32> scale_product = quant_val * weight_scale;
            fm_t final_result = scale_product * act_scale + bias_val;
            
            out_block[dim_offset] = final_result;
        }

        out_block = silu(out_block);
        out_stream << out_block;
    }
}

void write_out_stream(
    fm_block_t dst[],
    hls::stream<conv_out_t>& out_stream,
    unsigned int conv_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    static_assert(CONV_BLOCK_SIZE % FEATURE_BLOCK_SIZE == 0, "CONV_BLOCK_SIZE must be a multiple of FEATURE_BLOCK_SIZE");

    constexpr unsigned int num_blocks = CONV_BLOCK_SIZE / FEATURE_BLOCK_SIZE;
    constexpr unsigned int last_block = num_blocks - 1;

    unsigned int next_block = 0;
    unsigned int iters = num_patches * ceildiv(conv_dim, FEATURE_BLOCK_SIZE);

    conv_out_t buffer;

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1

        unsigned int block = next_block;
        next_block = (block == last_block) ? 0 : block + 1;

        if (block == 0)
        {
            out_stream >> buffer;
        }

        fm_block_t out_block;
        unsigned int base_idx = block * FEATURE_BLOCK_SIZE;

        FOR_EACH(j, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            out_block[j] = buffer[base_idx + j];
        }

        dst[i] = out_block;
    }
}

void compute_causal_conv_impl(
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_conv_weight_mag_block_t weights_mags[],
    const wt_conv_weight_sign_block_t weights_signs[],
    const wt_conv_bias_block_t bias[],
    const wt_conv_ws_block_t weight_scales[],
    unsigned int conv_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    #pragma HLS dataflow

    #pragma HLS aggregate variable=bias
    #pragma HLS aggregate variable=weight_scales

    hls::stream<conv_in_t> in_stream("conv_in_stream");
    #pragma HLS stream variable=in_stream depth=2
    hls::stream<conv_in_quant_t> in_quant_stream("conv_in_quant_stream");
    #pragma HLS stream variable=in_quant_stream depth=2
    hls::stream<conv_out_quant_t> out_quant_stream("conv_out_quant_stream");
    #pragma HLS stream variable=out_quant_stream depth=2
    hls::stream<conv_out_t> out_stream("conv_out_stream");
    #pragma HLS stream variable=out_stream depth=2

    hls::stream<wt_conv_as_t> act_scale_stream1("act_scale_stream1");
    #pragma HLS stream variable=act_scale_stream1 depth=2
    hls::stream<wt_conv_as_t> act_scale_stream2("act_scale_stream2");
    #pragma HLS stream variable=act_scale_stream2 depth=2

    read_in_stream(in_stream, act_scale_stream1, act_scale_stream2, src, conv_dim, num_patches);
    window_and_quantize_on_stream(in_quant_stream, in_stream, act_scale_stream1, conv_dim, num_patches);
    compute_conv_on_stream(out_quant_stream, in_quant_stream, weights_mags, weights_signs, conv_dim, num_patches);
    dequantize_on_stream(out_stream, out_quant_stream, bias, weight_scales, act_scale_stream2, conv_dim, num_patches);
    write_out_stream(dst, out_stream, conv_dim, num_patches);
}

void compute_causal_conv(
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_wide_t weights_mags_base[],
    const wt_wide_t weights_signs_base[],
    const wt_wide_t bias_base[],
    const wt_wide_t weight_scales_base[],
    unsigned int conv_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    
    static wt_conv_weight_mag_block_t weights_mags[ceildiv(D_INNER, CONV_BLOCK_SIZE)];
    static wt_conv_weight_sign_block_t weights_signs[ceildiv(D_INNER, CONV_BLOCK_SIZE)];
    static wt_conv_bias_block_t bias[ceildiv(D_INNER, CONV_BLOCK_SIZE)];
    static wt_conv_ws_block_t weight_scales[ceildiv(D_INNER, CONV_BLOCK_SIZE)];
    
    load_conv_weights_magnitudes(weights_mags, weights_mags_base, conv_dim);
    load_conv_weights_signs(weights_signs, weights_signs_base, conv_dim);
    load_conv_bias(bias, bias_base, conv_dim);
    load_conv_weight_scales(weight_scales, weight_scales_base, conv_dim);
    
    compute_causal_conv_impl(dst, src, weights_mags, weights_signs, bias, weight_scales, conv_dim, num_patches);
}

#endif // __INT_CONV_H__
