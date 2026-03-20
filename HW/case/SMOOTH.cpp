#include "../src/common.h"
#include "../src/smooth.h"
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
    const wt_wide_t smooth_scales_base[],
    unsigned int in_dim,
    unsigned int num_patches
)
{
    #pragma HLS interface ap_ctrl_chain port=return
    
    // AXI master interfaces
    #pragma HLS interface m_axi port=src bundle=in offset=slave depth=10000 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS interface m_axi port=dst bundle=out offset=slave depth=10000 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS interface m_axi port=smooth_scales_base bundle=weights offset=slave depth=256 max_read_burst_length=256
    
    // AXI-Lite control interface for scalars
    #pragma HLS interface s_axilite port=in_dim bundle=control
    #pragma HLS interface s_axilite port=num_patches bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    
    compute_smooth(dst, src, smooth_scales_base, in_dim, num_patches);
}

// ============================================================================
// Testbench code (for C simulation only, not synthesized)
// ============================================================================

#ifndef __SYNTHESIS__

// Test parameters
constexpr int LAYER_IDX = 0;

// Input/Output arrays
static patch_blocks_inner_t_fp REF_INPUT;
static patch_blocks_inner_t_fp REF_OUTPUT;
static patch_blocks_inner_t DUT_OUTPUT;

// Test function for the SMOOTH module
void test_smooth() {
    cout << "Testing SMOOTH module..." << endl;

    // Load input data - use x_proj input as smooth layer input
    cout << "Loading input data... " << flush;
    string input_file = string(get_project_root()) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.x_proj.input.float32.bin";
    if (!load_data(input_file, REF_INPUT)) {
        cout << "Failed to load input from " << input_file << endl;
        return;
    }
    
    // Convert float input to fixed-point
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_INNER; ++b) {
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                DUT_OUTPUT[p][b][o] = (fm_t)REF_INPUT[p][b][o];
            }
        }
    }
    cout << "done" << endl;

    // Load smooth scales into base array (for compute_smooth to load internally)
    cout << "Loading smooth scales... " << flush;
    string scale_file = string(get_project_root()) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.x_proj.smooth_scales.float32.bin";
    static wt_linear_ss_t ss_array[D_INNER];
    if (!load_data(scale_file, ss_array)) {
        cout << "Failed to load smooth scales from " << scale_file << endl;
        return;
    }
    
    // Convert smooth scales to 256-bit wide format for burst transfer
    // Each 256-bit word holds 8 elements (32 bits each)
    constexpr unsigned int SMOOTH_SCALES_PER_WORD = AXI_XFER_BIT_WIDTH / 32;
    constexpr unsigned int scale_words = ceildiv(D_INNER, SMOOTH_SCALES_PER_WORD);
    static wt_wide_t smooth_scales_base[scale_words];
    
    // Pack scales into 256-bit words
    for (unsigned int i = 0; i < scale_words; ++i) {
        wt_wide_t word = 0;
        for (unsigned int j = 0; j < SMOOTH_SCALES_PER_WORD; ++j) {
            unsigned int idx = i * SMOOTH_SCALES_PER_WORD + j;
            if (idx < D_INNER) {
                ap_uint<32> bits = *reinterpret_cast<ap_uint<32>*>(&ss_array[idx]);
                word.range(j * 32 + 31, j * 32) = bits;
            }
        }
        smooth_scales_base[i] = word;
    }
    cout << "done" << endl;
    
    cout << "Running SMOOTH operation... " << flush;
    top(
        reinterpret_cast<fm_block_t*>(DUT_OUTPUT),
        reinterpret_cast<const fm_block_t*>(DUT_OUTPUT),
        smooth_scales_base,
        D_INNER,
        NUM_PATCHES
    );
    cout << "done" << endl;

    cout << "Loading reference output... " << flush;
    string ref_file = string(get_project_root()) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.x_proj.input.float32.bin";
    if (!load_data(ref_file, REF_OUTPUT)) {
        cout << "Failed to load reference output from " << ref_file << endl;
        return;
    }
    
    // Apply smooth scales to reference to get expected output
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_INNER; ++b) {
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                unsigned int dim = b * FEATURE_BLOCK_SIZE + o;
                if (dim < D_INNER) {
                    REF_OUTPUT[p][b][o] *= (float)ss_array[dim];
                }
            }
        }
    }
    cout << "done" << endl;
    
    // Compare results
    cout << "Comparing results..." << endl;
    perform_complete_comparison(
        DUT_OUTPUT,
        REF_OUTPUT,
        "SMOOTH output",
        "Smooth Layer Output",
        NUM_PATCHES,
        D_INNER,
        CLS_TOKEN_IDX
    );
}

// Main function to run the SMOOTH test
int main() {
    test_smooth();
    cout << "SMOOTH test completed" << endl;
    return 0;
}

#endif // __SYNTHESIS__
