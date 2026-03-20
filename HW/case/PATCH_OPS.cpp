#include "../src/common.h"
#include "../src/patch_ops.h"
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
    unsigned int mode,
    unsigned int num_patches,
    unsigned int cls_token_idx,
    unsigned int inner_dim,
    unsigned int model_dim
)
{
    #pragma HLS interface ap_ctrl_chain port=return
    
    // AXI master interfaces
    #pragma HLS interface m_axi port=src bundle=in offset=slave depth=10000 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS interface m_axi port=dst bundle=out offset=slave depth=10000 max_read_burst_length=256 max_write_burst_length=256
    
    // AXI-Lite control interface for scalars
    #pragma HLS interface s_axilite port=mode bundle=control
    #pragma HLS interface s_axilite port=num_patches bundle=control
    #pragma HLS interface s_axilite port=cls_token_idx bundle=control
    #pragma HLS interface s_axilite port=inner_dim bundle=control
    #pragma HLS interface s_axilite port=model_dim bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    
    patch_ops_impl(dst, src, (PatchOpMode)mode, num_patches, cls_token_idx, inner_dim, model_dim);
}

// ============================================================================
// Testbench code (for C simulation only, not synthesized)
// ============================================================================

#ifndef __SYNTHESIS__

// Test parameters
constexpr int LAYER_IDX = 0;

// Test function for FLIP_PATCH mode
void test_flip_patch() {
    cout << "Testing patch_ops (FLIP_PATCH mode)..." << endl;
    
    // Use flat arrays like LINEAR
    static fm_block_t INPUT_COPY[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
    static fm_block_t DUT_OUTPUT[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
    static patch_blocks_inner_t_fp ref_output;
    
    // Initialize input with patch index as a marker
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_INNER; ++b) {
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                unsigned int flat_idx = p * NUM_FEATURE_BLOCKS_INNER + b;
                INPUT_COPY[flat_idx][o] = (fm_t)(p * 0.1f);
            }
        }
    }
    
    // Run flip_patch via top()
    cout << "Running patch_ops (FLIP mode)... " << flush;
    top(DUT_OUTPUT, INPUT_COPY, PATCH_OP_FLIP, NUM_PATCHES, CLS_TOKEN_IDX, D_INNER, D_MODEL);
    cout << "done" << endl;
    
    // Create reference: patches should be in reverse order
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        unsigned int flipped_patch = NUM_PATCHES - 1 - p;
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_INNER; ++b) {
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                unsigned int flat_idx = flipped_patch * NUM_FEATURE_BLOCKS_INNER + b;
                ref_output[p][b][o] = (float)INPUT_COPY[flat_idx][o];
            }
        }
    }
    
    // Compare results
    cout << "Comparing results..." << endl;
    perform_complete_comparison(
        reinterpret_cast<const patch_blocks_inner_t&>(DUT_OUTPUT),
        ref_output,
        "patch_ops (FLIP) output",
        "Flipped Patches",
        NUM_PATCHES,
        D_INNER,
        CLS_TOKEN_IDX
    );
}

// Test function for LOAD_CLS_TOKEN mode
void test_load_cls_token() {
    cout << "Testing patch_ops (LOAD_CLS_TOKEN mode)..." << endl;

    // Load input data (patch blocks)
    cout << "Loading input data... " << flush;
    static patch_blocks_t_fp input_fp;
    string input_file = string(get_project_root()) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".norm.output.float32.bin";
    if (!load_data(input_file, input_fp)) {
        cout << "Failed to load input from " << input_file << endl;
        return;
    }
    cout << "done" << endl;
    
    // Convert to fixed-point - use flat arrays
    static fm_block_t INPUT_COPY[NUM_PATCHES * NUM_FEATURE_BLOCKS];
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS; ++b) {
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                unsigned int flat_idx = p * NUM_FEATURE_BLOCKS + b;
                INPUT_COPY[flat_idx][o] = (fm_t)input_fp[p][b][o];
            }
        }
    }
    
    // Extract CLS token - use flat array
    // Make it match depth parameter to avoid AXI write bounds issues
    static fm_block_t DUT_OUTPUT[10000];
    cout << "Running patch_ops (LOAD_CLS mode)... " << flush;
    top(DUT_OUTPUT, INPUT_COPY, PATCH_OP_LOAD_CLS, NUM_PATCHES, CLS_TOKEN_IDX, D_INNER, D_MODEL);
    cout << "done" << endl;
    
    // Create reference: CLS token should be the same as input[CLS_TOKEN_IDX]
    fm_blocks_t_fp ref_cls_token[1];
    for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS; ++b) {
        for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
            ref_cls_token[0][b][o] = input_fp[CLS_TOKEN_IDX][b][o];
        }
    }
    
    // Compare results (CLS token is a single patch)
    cout << "Comparing results..." << endl;
    float mse = 0.0f;
    float mae = 0.0f;
    unsigned int total_dims = 0;
    
    for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS; ++b) {
        for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
            unsigned int dim = b * FEATURE_BLOCK_SIZE + o;
            if (dim >= D_MODEL) break;
            
            float computed_val = DUT_OUTPUT[b][o].to_float();
            float ref_val = ref_cls_token[0][b][o];
            float error = ref_val - computed_val;
            float abs_error = (error < 0.0f) ? -error : error;
            
            mse += error * error;
            mae += abs_error;
            total_dims++;
        }
    }
    
    mse /= static_cast<float>(total_dims);
    mae /= static_cast<float>(total_dims);
    
    cout << "CLS Token Extraction:" << endl;
    cout << "  MSE: " << mse << endl;
    cout << "  MAE: " << mae << endl;
    cout << endl;
}

// Main function to run the patch_ops tests
int main() {
    test_flip_patch();
    cout << "\nFLIP_PATCH test completed" << endl;
    
    test_load_cls_token();
    cout << "\nLOAD_CLS_TOKEN test completed" << endl;
    
    cout << "\nPATCH_OPS test completed" << endl;
    return 0;
}

#endif // __SYNTHESIS__

