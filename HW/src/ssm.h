#ifndef __INT_SSM_H__
#define __INT_SSM_H__

#include "common.h"
#include <hls_stream.h>
#include <hls_math.h>

// ============================================================================
// FAST EXPONENTIAL
// ============================================================================

inline fm_t fast_exp_approx(fm_t x) {
    #pragma HLS inline
    // exp(x) ≈ (1 + x/32)^32 for x < 0
    // This is always positive and much more stable than Taylor for large negative x
    fm_t term = fm_t(1.0) + x * fm_t(0.03125);
    if (term < fm_t(0.0)) term = fm_t(0.0);
    
    fm_t t2 = term * term;
    fm_t t4 = t2 * t2;
    fm_t t8 = t4 * t4;
    fm_t t16 = t8 * t8;
    fm_t res = t16 * t16;
    
    return res;
}

typedef gate_block_t prescan_gate_out_t;
typedef token_block_t prescan_token_out_t;

// Load A buffer
void load_A_buffer(
    fm_block_t dst[],
    const wt_wide_t src[],
    unsigned int scan_dim
)
{
    #pragma HLS inline off
    
    unsigned int num_blocks = ceildiv(scan_dim, FEATURE_BLOCK_SIZE);
    unsigned int num_words = num_blocks;
    
    FOR_EACH(i, num_words)
    {
        #pragma HLS pipeline II=1
        wt_wide_t word = src[i];
        
        FOR_EACH(j, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            unsigned int block_idx = i;
            unsigned int elem_idx = block_idx * FEATURE_BLOCK_SIZE + j;
            if (elem_idx < scan_dim) {
                ap_uint<32> elem_bits = word.range(j * 32 + 31, j * 32);
                dst[block_idx][j].range(31, 0) = elem_bits;
            } else {
                dst[block_idx][j] = (fm_t)0;
            }
        }
    }
}

// Load D buffer
void load_D_buffer(
    fm_block_t dst[],
    const wt_wide_t src[],
    unsigned int inner_dim
)
{
    #pragma HLS inline off
    
    unsigned int num_blocks = ceildiv(inner_dim, FEATURE_BLOCK_SIZE);
    unsigned int num_words = num_blocks;
    
    FOR_EACH(i, num_words)
    {
        #pragma HLS pipeline II=1
        wt_wide_t word = src[i];
        
        FOR_EACH(j, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            unsigned int block_idx = i;
            unsigned int elem_idx = block_idx * FEATURE_BLOCK_SIZE + j;
            if (elem_idx < inner_dim) {
                ap_uint<32> elem_bits = word.range(j * 32 + 31, j * 32);
                dst[block_idx][j].range(31, 0) = elem_bits;
            } else {
                dst[block_idx][j] = (fm_t)0;
            }
        }
    }
}

// Optimized unified pipeline: Prescan + Scan in single loop
void compute_state_update(
    hls::stream<scan_out_block_t>& out_scan_stream,
    hls::stream<fm_block_t>& in_delta_stream,
    hls::stream<fm_block_t>& in_u_stream,
    hls::stream<fm_block_t>& in_B_stream,
    const fm_block_t A_src[],
    unsigned int scan_dim,
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    token_t pe_state_token[NUM_FEATURE_BLOCKS_SCAN][FEATURE_BLOCK_SIZE];
    #pragma HLS array_partition variable=pe_state_token complete dim=2

    // Initialize state
    INIT_LOOP: FOR_EACH(block, ceildiv(scan_dim, FEATURE_BLOCK_SIZE)) {
        #pragma HLS unroll
        FOR_EACH(pe, FEATURE_BLOCK_SIZE) {
            #pragma HLS unroll
            pe_state_token[block][pe] = token_t(0);
        }
    }
    
    constexpr unsigned int state_dim_iters = ceildiv(D_STATE, FEATURE_BLOCK_SIZE);
    unsigned int last_state_dim_iter = state_dim_iters - 1;
    unsigned int inner_dim_iters = inner_dim;
    unsigned int total_dim_iters = inner_dim * state_dim_iters;
    unsigned int last_total_dim_iter = total_dim_iters - 1;

    unsigned int next_state_dim_block = 0;
    unsigned int next_inner_dim_block = 0;
    unsigned int next_total_dim_block = 0;

    fm_block_t B_blocks[NUM_FEATURE_BLOCKS_STATE];
    #pragma HLS array_partition variable=B_blocks complete

    fm_block_t delta_block;
    fm_block_t u_block;

    prescan_gate_out_t delta_A_block;
    prescan_token_out_t delta_Bu_block;

    unsigned int total_iters = num_patches * total_dim_iters;
    unsigned int read_iters = FEATURE_BLOCK_SIZE * state_dim_iters;

    UNIFIED_LOOP: FOR_EACH(i, total_iters)
    {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=pe_state_token inter false

        unsigned int total_dim_block = next_total_dim_block;
        next_total_dim_block = (total_dim_block == last_total_dim_iter) ? 0 : total_dim_block + 1;

        unsigned int state_dim_block = next_state_dim_block;
        next_state_dim_block = (state_dim_block == last_state_dim_iter) ? 0 : state_dim_block + 1;

        unsigned int inner_dim_block = next_inner_dim_block;
        next_inner_dim_block = (total_dim_block == last_total_dim_iter)
        ? 0 : (state_dim_block == last_state_dim_iter) ? inner_dim_block + 1 : inner_dim_block;

        unsigned int inner_dim_offset = inner_dim_block % FEATURE_BLOCK_SIZE;

        if (inner_dim_block == 0) {
            in_B_stream >> B_blocks[state_dim_block];
        }
        if (total_dim_block % read_iters == 0) {
            in_delta_stream >> delta_block;
            in_u_stream >> u_block;
        }

        fm_block_t A_slice = A_src[total_dim_block];
        fm_block_t B_slice = B_blocks[state_dim_block];

        // Prescan calculation
        fm_t delta_val = delta_block[inner_dim_offset];
        fm_t u_val = u_block[inner_dim_offset];
        fm_t delta_u = delta_val * u_val;
        FOR_EACH(state, FEATURE_BLOCK_SIZE)
        {
            delta_A_block[state] = delta_val * A_slice[state];
            delta_A_block[state] = fast_exp_approx(delta_A_block[state]);
            delta_Bu_block[state] = delta_u * B_slice[state];
        }

        // Scan calculation - integrated to avoid buffering
        scan_out_block_t output_block;
        PE_LOOP: FOR_EACH(pe, FEATURE_BLOCK_SIZE) {
            #pragma HLS unroll factor=4
            
            token_t current_token = pe_state_token[total_dim_block][pe];
            token_t new_token = current_token * delta_A_block[pe] + delta_Bu_block[pe];
            
            pe_state_token[total_dim_block][pe] = new_token;
            output_block[pe] = scan_t(new_token);
        }
        
        out_scan_stream << output_block;
    }
}

// Optimized x_C computation with reduced buffering
void state_to_xC_stream(
    hls::stream<scan_out_block_t>& in_x_C_stream,
    hls::stream<scan_out_block_t>& out_scan_stream,
    hls::stream<fm_block_t>& in_C_stream,
    unsigned int scan_dim,
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    constexpr unsigned int state_dim_iters = ceildiv(D_STATE, FEATURE_BLOCK_SIZE);
    unsigned int last_state_dim_iter = state_dim_iters - 1;
    unsigned int total_dim_iters = inner_dim * state_dim_iters;
    unsigned int last_total_dim_iter = total_dim_iters - 1;

    unsigned int next_state_dim_block = 0;
    unsigned int next_inner_dim_block = 0;
    unsigned int next_total_dim_block = 0;
    unsigned int patch_idx = 0;

    fm_block_t C_blocks[NUM_FEATURE_BLOCKS_STATE];
    #pragma HLS array_partition variable=C_blocks complete

    scan_t tmp_res;
    scan_out_block_t res_block;
    scan_t tmp_res_reg = 0;
    scan_t partial_products[FEATURE_BLOCK_SIZE];
    #pragma HLS array_partition variable=partial_products complete

    unsigned int iters = num_patches * total_dim_iters;

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline II=1

        scan_out_block_t x_block;
        out_scan_stream >> x_block;

        unsigned int total_dim_block = next_total_dim_block;
        next_total_dim_block = (total_dim_block == last_total_dim_iter) ? 0 : total_dim_block + 1;

        unsigned int state_dim_block = next_state_dim_block;
        next_state_dim_block = (state_dim_block == last_state_dim_iter) ? 0 : state_dim_block + 1;

        unsigned int inner_dim_block = next_inner_dim_block;
        next_inner_dim_block = (total_dim_block == last_total_dim_iter)
        ? 0 : (state_dim_block == last_state_dim_iter) ? inner_dim_block + 1 : inner_dim_block;

        unsigned int block = inner_dim_block % FEATURE_BLOCK_SIZE;

        tmp_res = (state_dim_block == 0) ? (scan_t)0 : tmp_res_reg;

        if (inner_dim_block == 0) {
            in_C_stream >> C_blocks[state_dim_block];
        }

        // Tree reduction for better timing
        FOR_EACH(dim, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            partial_products[dim] = x_block[dim] * C_blocks[state_dim_block][dim];
        }
        
        scan_t s01 = partial_products[0] + partial_products[1];
        scan_t s23 = partial_products[2] + partial_products[3];
        scan_t s45 = partial_products[4] + partial_products[5];
        scan_t s67 = partial_products[6] + partial_products[7];
        scan_t s03 = s01 + s23;
        scan_t s47 = s45 + s67;
        scan_t current_sum = s03 + s47;
        
        tmp_res += current_sum;
        tmp_res_reg = tmp_res;
        res_block[block] = tmp_res;

        if (block == FEATURE_BLOCK_SIZE - 1 && state_dim_block == last_state_dim_iter) {
            in_x_C_stream << res_block;
        }

        if (total_dim_block == last_total_dim_iter) {
            patch_idx++;
        }
    }
}

// Read delta stream
void read_in_delta_stream(
    hls::stream<fm_block_t>& in_delta_stream,
    const fm_block_t delta_src[],
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    FOR_EACH(i, num_patches * ceildiv(inner_dim, FEATURE_BLOCK_SIZE))
    {
        #pragma HLS pipeline

        fm_block_t delta_block = delta_src[i];
        in_delta_stream << delta_block;
    }
}

// Read B stream
void read_in_B_stream(
    hls::stream<fm_block_t>& in_B_stream,
    const fm_block_t B_src[],
    unsigned int scan_dim,
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    FOR_EACH(i, num_patches * ceildiv(D_STATE, FEATURE_BLOCK_SIZE))
    {
        #pragma HLS pipeline

        fm_block_t block = B_src[i];
        in_B_stream << block;
    }
}

void read_in_C_stream(
    hls::stream<fm_block_t>& in_C_stream,
    const fm_block_t C_src[],
    unsigned int scan_dim,
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    FOR_EACH(i, num_patches * ceildiv(D_STATE, FEATURE_BLOCK_SIZE))
    {
        #pragma HLS pipeline

        fm_block_t block = C_src[i];
        in_C_stream << block;
    }
}

void read_u_streams(
    hls::stream<fm_block_t>& u_for_prescan_stream,
    hls::stream<fm_block_t>& u_for_output_stream,
    const fm_block_t u_src[],
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    
    FOR_EACH(i, num_patches * ceildiv(inner_dim, FEATURE_BLOCK_SIZE))
    {
        #pragma HLS pipeline II=1
        
        fm_block_t u_block = u_src[i];
        
        // Send to both streams
        u_for_prescan_stream << u_block;
        u_for_output_stream << u_block;
    }
}

// Optimized u*D multiplication with reduced stream depth
void read_in_u_D_stream(
    hls::stream<fm_block_t>& in_u_d_stream,
    hls::stream<fm_block_t>& u_stream,
    const fm_block_t D_src[],
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    unsigned int next_D_src_block = 0;
    unsigned int num_D_blocks = ceildiv(inner_dim, FEATURE_BLOCK_SIZE);

    FOR_EACH(i, num_patches * num_D_blocks)
    {
        #pragma HLS pipeline II=1

        unsigned int D_src_block = next_D_src_block;
        next_D_src_block = (D_src_block == num_D_blocks - 1) ? 0 : D_src_block + 1;

        fm_block_t u_block;
        u_stream >> u_block;
        
        fm_block_t D_slice = D_src[D_src_block];

        fm_block_t block_for_dst;
        FOR_EACH(j, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            block_for_dst[j] = u_block[j] * D_slice[j];
        }
        in_u_d_stream << block_for_dst;
    }
}

// Read z_silu - same as original
void read_in_z_silu_stream(
    hls::stream<fm_block_t>& in_z_silu_stream,
    const fm_block_t z_silu_src[],
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    FOR_EACH(i, num_patches * ceildiv(inner_dim, FEATURE_BLOCK_SIZE))
    {
        #pragma HLS pipeline

        fm_block_t z_silu_block = z_silu_src[i];
        in_z_silu_stream << z_silu_block;
    }
}

// Optimized final output computation
void compute_output_on_stream(
    hls::stream<scan_out_block_t>& out_stream,
    hls::stream<scan_out_block_t>& in_x_C_stream,
    hls::stream<fm_block_t>& in_u_d_stream,
    hls::stream<fm_block_t>& in_z_silu_stream,
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    unsigned int iters = num_patches * ceildiv(inner_dim, FEATURE_BLOCK_SIZE);

    scan_out_block_t x_C_block;
    fm_block_t u_d_block;
    fm_block_t z_silu_block;

    FOR_EACH(i, iters)
    {
        #pragma HLS pipeline

        in_x_C_stream >> x_C_block;
        in_u_d_stream >> u_d_block;
        in_z_silu_stream >> z_silu_block;

        scan_out_block_t out_block;
        FOR_EACH(dim, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS unroll
            out_block[dim] = (x_C_block[dim] + u_d_block[dim]) * z_silu_block[dim];
        }
        out_stream << out_block;
    }
}

// Write output - same as original
void write_output_stream(
    scan_out_block_t dst[],
    hls::stream<scan_out_block_t>& out_stream,
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off

    FOR_EACH(i, num_patches * ceildiv(inner_dim, FEATURE_BLOCK_SIZE))
    {
        #pragma HLS pipeline

        scan_out_block_t stream_block;
        out_stream >> stream_block;
        dst[i] = stream_block;
    }
}

// Optimized main implementation with integrated prescan+scan
void compute_ssm_output_impl(
    scan_out_block_t out[],
    const fm_block_t u_src[],
    const fm_block_t delta_src[],
    const fm_block_t z_silu_src[],
    const fm_block_t A_src[],
    const fm_block_t B_src[],
    const fm_block_t C_src[],
    const fm_block_t D_src[],
    unsigned int scan_dim,
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    #pragma HLS dataflow
    
    // Streams for unified prescan+scan stage
    hls::stream<fm_block_t> in_delta_stream("in_delta_stream");
    #pragma HLS stream variable=in_delta_stream depth=2

    hls::stream<fm_block_t> u_for_prescan_stream("u_for_prescan_stream");
    #pragma HLS stream variable=u_for_prescan_stream depth=2

    hls::stream<fm_block_t> u_for_output_stream("u_for_output_stream");
    #pragma HLS stream variable=u_for_output_stream depth=2

    hls::stream<fm_block_t> in_B_stream("in_B_stream");
    #pragma HLS stream variable=in_B_stream depth=2

    hls::stream<fm_block_t> in_C_stream("in_C_stream");
    #pragma HLS stream variable=in_C_stream depth=2

    // Single stream from unified prescan+scan - drastically reduced size
    hls::stream<scan_out_block_t> out_scan_stream("out_scan_stream");
    #pragma HLS stream variable=out_scan_stream depth=2

    // Streams for output stage
    hls::stream<scan_out_block_t> in_x_C_stream("in_x_C_stream");
    #pragma HLS stream variable=in_x_C_stream depth=2

    hls::stream<fm_block_t> in_u_d_stream("in_u_d_stream");
    #pragma HLS stream variable=in_u_d_stream depth=2

    hls::stream<fm_block_t> in_z_silu_stream("in_z_silu_stream");
    #pragma HLS stream variable=in_z_silu_stream depth=2

    hls::stream<scan_out_block_t> out_stream("out_stream");
    #pragma HLS stream variable=out_stream depth=2

    // Input reading
    read_in_delta_stream(in_delta_stream, delta_src, inner_dim, num_patches);
    read_u_streams(u_for_prescan_stream, u_for_output_stream, u_src, inner_dim, num_patches);
    read_in_B_stream(in_B_stream, B_src, scan_dim, inner_dim, num_patches);
    read_in_C_stream(in_C_stream, C_src, scan_dim, inner_dim, num_patches);
    read_in_z_silu_stream(in_z_silu_stream, z_silu_src, inner_dim, num_patches);
    
    // Fine-grained SSM computation
    compute_state_update(out_scan_stream, in_delta_stream, u_for_prescan_stream, in_B_stream, A_src, scan_dim, inner_dim, num_patches);
    state_to_xC_stream(in_x_C_stream, out_scan_stream, in_C_stream, scan_dim, inner_dim, num_patches);
    read_in_u_D_stream(in_u_d_stream, u_for_output_stream, D_src, inner_dim, num_patches);
    compute_output_on_stream(out_stream, in_x_C_stream, in_u_d_stream, in_z_silu_stream, inner_dim, num_patches);
    write_output_stream(out, out_stream, inner_dim, num_patches);
}

void compute_ssm_output(
    scan_out_block_t out[],
    const fm_block_t u_src[],
    const fm_block_t delta_src[],
    const fm_block_t z_silu_src[],
    const wt_wide_t A_base[],
    const fm_block_t B_src[],
    const fm_block_t C_src[],
    const wt_wide_t D_base[],
    unsigned int scan_dim,
    unsigned int inner_dim,
    unsigned int num_patches
)
{
    #pragma HLS inline off
    
    // Local buffers
    fm_block_t A_buffer[NUM_FEATURE_BLOCKS_SCAN];
    fm_block_t D_buffer[NUM_FEATURE_BLOCKS_INNER];
    
    // Load A and D from base arrays
    load_A_buffer(A_buffer, A_base, scan_dim);
    load_D_buffer(D_buffer, D_base, inner_dim);
    
    compute_ssm_output_impl(out, u_src, delta_src, z_silu_src, A_buffer, B_src, C_src, D_buffer, scan_dim, inner_dim, num_patches);
}

#endif // __INT_SCAN_O_H__
