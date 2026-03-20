#include "../src/common.h"
#include "../src/conv.h"
#include "../src/utils.h"
#include "../testbench/data_loader.hpp"
#include "../testbench/tb_utils.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

using std::cout;
using std::endl;
using std::flush;
using std::string;

// ============================================================================
// Top-level function
// ============================================================================
void top(
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
    #pragma HLS interface ap_ctrl_chain port=return
    
    #pragma HLS interface m_axi port=src bundle=in_r offset=slave depth=10000 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS interface m_axi port=dst bundle=out_r offset=slave depth=10000 max_read_burst_length=256 max_write_burst_length=256
    
    #pragma HLS interface m_axi port=weights_mags_base bundle=weights offset=slave depth=4096 max_read_burst_length=256
    #pragma HLS interface m_axi port=weights_signs_base bundle=weights offset=slave depth=4096 max_read_burst_length=256
    #pragma HLS interface m_axi port=bias_base bundle=weights offset=slave depth=256 max_read_burst_length=256
    #pragma HLS interface m_axi port=weight_scales_base bundle=weights offset=slave depth=256 max_read_burst_length=256
    
    #pragma HLS interface s_axilite port=conv_dim bundle=control
    #pragma HLS interface s_axilite port=num_patches bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    
    compute_causal_conv(dst, src, weights_mags_base, weights_signs_base, bias_base, weight_scales_base, conv_dim, num_patches);
}

// ============================================================================
// Testbench code (for C simulation only, not synthesized)
// ============================================================================

#ifndef __SYNTHESIS__

// Test parameters
constexpr int CONV_DIM = D_INNER;
constexpr int LAYER_IDX = 0;

// Input/Output arrays
static patch_blocks_inner_t_fp REF_INPUT;
static patch_blocks_inner_t_fp REF_OUTPUT;
static fm_block_t DUT_OUTPUT[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
static fm_block_t INPUT_COPY[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];

void test_conv() {
    cout << "Testing CONV (conv1d) module..." << endl;

    cout << "Loading input data... " << flush;
    string input_file = string(get_project_root()) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.in_proj.output.x.float32.bin";
    if (!load_data(input_file, REF_INPUT)) {
        cout << "Failed to load input from " << input_file << endl;
        return;
    }

    // Convert float input to fixed-point, using flat array indexing like LINEAR
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_INNER; ++b) {
            unsigned int flat_idx = p * NUM_FEATURE_BLOCKS_INNER + b;
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                DUT_OUTPUT[flat_idx][o] = (fm_t)REF_INPUT[p][b][o];
            }
        }
    }
    cout << "done" << endl;

    cout << "Loading weights... " << flush;
    const char* project_root = get_project_root();
    string mag_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.conv1d.weight.magnitude.bin";
    string sign_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.conv1d.weight.sign.bin";
    string bias_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.conv1d.bias.float32.bin";
    string scale_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.conv1d.weight_scale.float32.bin";
    
    // Load all in_proj weights - use direct calculation like LINEAR for compile-time array size
    constexpr unsigned int total_elems = CONV_KERNEL_SIZE * D_INNER;
    static wt_wide_t weights_mags_base[ceildiv(CONV_KERNEL_SIZE * D_INNER, MAGS_PER_WORD)];
    static wt_wide_t weights_signs_base[ceildiv(CONV_KERNEL_SIZE * D_INNER, SIGNS_PER_WORD)];
    
    if (!load_int4_file_to_burst_words(mag_file, weights_mags_base, total_elems)) {
        cout << "Failed to load weight magnitudes" << endl;
        return;
    }
    if (!load_bit_file_to_burst_words(sign_file, weights_signs_base, total_elems)) {
        cout << "Failed to load weight signs" << endl;
        return;
    }
    
    static wt_conv_bias_t bias_array[D_INNER];
    if (!load_data(bias_file, bias_array)) {
        cout << "Failed to load bias" << endl;
        return;
    }
    static wt_wide_t bias_base[ceildiv(D_INNER, 8U)];
    for (unsigned int i = 0; i < ceildiv(D_INNER, 8U); ++i) {
        wt_wide_t word = 0;
        for (unsigned int j = 0; j < 8; ++j) {
            unsigned int idx = i * 8 + j;
            if (idx < D_INNER) {
                ap_uint<32> elem_bits = *reinterpret_cast<ap_uint<32>*>(&bias_array[idx]);
                word.range(j * 32 + 31, j * 32) = elem_bits;
            }
        }
        bias_base[i] = word;
    }
    
    static wt_conv_ws_t ws_array[D_INNER];
    if (!load_data(scale_file, ws_array)) {
        cout << "Failed to load weight scales" << endl;
        return;
    }
    static wt_wide_t weight_scales_base[ceildiv(D_INNER, 8U)];
    for (unsigned int i = 0; i < ceildiv(D_INNER, 8U); ++i) {
        wt_wide_t word = 0;
        for (unsigned int j = 0; j < 8; ++j) {
            unsigned int idx = i * 8 + j;
            if (idx < D_INNER) {
                ap_uint<32> elem_bits = *reinterpret_cast<ap_uint<32>*>(&ws_array[idx]);
                word.range(j * 32 + 31, j * 32) = elem_bits;
            }
        }
        weight_scales_base[i] = word;
    }
    cout << "done" << endl;
    
    // Copy input for in-place operation, using flat array indexing like LINEAR
    for (unsigned int i = 0; i < NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER; ++i) {
        for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
            INPUT_COPY[i][o] = DUT_OUTPUT[i][o];
        }
    }
    
    cout << "Running CONV operation... " << flush;
    top(
        DUT_OUTPUT,
        INPUT_COPY,
        weights_mags_base,
        weights_signs_base,
        bias_base,
        weight_scales_base,
        CONV_DIM,
        NUM_PATCHES
    );
    cout << "done" << endl;

    cout << "Loading reference output... " << flush;
    string ref_file = string(project_root) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.x_proj.input.float32.bin";
    if (!load_data(ref_file, REF_OUTPUT)) {
        cout << "Failed to load reference output from " << ref_file << endl;
        return;
    }
    cout << "done" << endl;
    
    // Compare results - cast flat array to 3D type for comparison function (like LINEAR)
    cout << "Comparing results..." << endl;
    perform_complete_comparison(
        reinterpret_cast<const patch_blocks_inner_t&>(DUT_OUTPUT),
        REF_OUTPUT,
        "CONV (conv1d) output",
        "Conv1d Output",
        NUM_PATCHES,
        D_INNER,
        CLS_TOKEN_IDX
    );
}

int main() {
    test_conv();
    cout << "CONV test completed" << endl;
    return 0;
}

#endif // __SYNTHESIS__
