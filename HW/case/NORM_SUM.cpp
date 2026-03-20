#include "../src/common.h"
#include "../src/norm_sum.h"
#include "../testbench/tb_utils.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using std::cout;
using std::endl;

// ============================================================================
// Top-level function
// ============================================================================
void top(
    fm_block_t dst[],
    const fm_block_t src_a[],
    const fm_block_t src_b[],
    const wt_wide_t weights_base[],
    unsigned int model_dim,
    unsigned int num_patches,
    NormSumMode mode
)
{
    #pragma HLS interface ap_ctrl_chain port=return
    
    #pragma HLS interface m_axi port=src_a bundle=in_a offset=slave depth=5000 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS interface m_axi port=src_b bundle=in_b offset=slave depth=5000 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS interface m_axi port=dst bundle=out_r offset=slave depth=5000 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS interface m_axi port=weights_base bundle=weights offset=slave depth=128 max_read_burst_length=256
    
    #pragma HLS interface s_axilite port=model_dim bundle=control
    #pragma HLS interface s_axilite port=num_patches bundle=control
    #pragma HLS interface s_axilite port=mode bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    
    norm_sum_impl(dst, src_a, src_b, weights_base, model_dim, num_patches, mode);
}

// ============================================================================
// Testbench code (for C simulation only, not synthesized)
// ============================================================================

#ifndef __SYNTHESIS__

void test_norm_sum() {
    unsigned int model_dim = D_MODEL;
    unsigned int num_patches = NUM_PATCHES;
    unsigned int num_blocks = (model_dim / FEATURE_BLOCK_SIZE) * num_patches;
    unsigned int num_weight_words = ceildiv(model_dim, 16U);
    
    // Use raw arrays for COSIM stability
    static fm_block_t src_a[5000];
    static fm_block_t src_b[5000];
    static fm_block_t dst[5000];
    static wt_wide_t weights[128];

    // Initialize inputs
    for (int i = 0; i < num_blocks; i++) {
        for (int j = 0; j < FEATURE_BLOCK_SIZE; j++) {
            src_a[i][j] = (fm_t)((i * FEATURE_BLOCK_SIZE + j) % 10) * (fm_t)0.1;
            src_b[i][j] = (fm_t)((i * FEATURE_BLOCK_SIZE + j + 5) % 10) * (fm_t)0.1;
        }
    }

    for (int i = 0; i < num_weight_words; i++) {
        wt_wide_t word = 0;
        for (int j = 0; j < 16; j++) {
            unsigned int idx = i * 16 + j;
            if (idx < model_dim) {
                wt_norm_t w = (wt_norm_t)0.5;
                ap_uint<16> bits = w.range(15, 0);
                word.range(j * 16 + 15, j * 16) = bits;
            }
        }
        weights[i] = word;
    }

    // Test all modes
    NormSumMode modes[] = {NORM_SUM_BOTH, NORM_SUM_NORM_ONLY, NORM_SUM_ADD_ONLY, NORM_SUM_DIV2_ONLY};
    const char* mode_names[] = {"BOTH", "NORM_ONLY", "ADD_ONLY", "DIV2_ONLY"};

    bool all_passed = true;

    for (int m = 0; m < 4; m++) {
        NormSumMode test_mode = modes[m];
        std::cout << "Testing Mode: " << mode_names[m] << std::endl;

        // Run HLS top function
        top(dst, src_a, src_b, weights, model_dim, num_patches, test_mode);

        // Reference calculation and verification
        double max_error = 0;
        double total_sq_error = 0;
        int count = 0;

        for (int p = 0; p < num_patches; p++) {
            double sum_sq = 1e-5; // epsilon
            std::vector<double> patch_sum(model_dim);

            // First pass: compute sum/norm
            for (int b = 0; b < model_dim / FEATURE_BLOCK_SIZE; b++) {
                int block_idx = p * (model_dim / FEATURE_BLOCK_SIZE) + b;
                for (int i = 0; i < FEATURE_BLOCK_SIZE; i++) {
                    double val_a = (double)src_a[block_idx][i];
                    double val_b = (double)src_b[block_idx][i];
                    double val;
                    if (test_mode == NORM_SUM_NORM_ONLY) {
                        val = val_a;
                    } else {
                        val = val_a + val_b;
                    }
                    patch_sum[b * FEATURE_BLOCK_SIZE + i] = val;
                    sum_sq += val * val / model_dim;
                }
            }

            double rms_inv = 1.0 / std::sqrt(sum_sq);

            // Second pass: compute output and compare
            for (int b = 0; b < model_dim / FEATURE_BLOCK_SIZE; b++) {
                int block_idx = p * (model_dim / FEATURE_BLOCK_SIZE) + b;
                for (int i = 0; i < FEATURE_BLOCK_SIZE; i++) {
                    double ref;
                    double val = patch_sum[b * FEATURE_BLOCK_SIZE + i];
                    
                    if (test_mode == NORM_SUM_BOTH) {
                        ref = val * rms_inv * 0.5; // weight=0.5
                    } else if (test_mode == NORM_SUM_NORM_ONLY) {
                        ref = val * rms_inv * 0.5; // weight=0.5
                    } else if (test_mode == NORM_SUM_ADD_ONLY) {
                        ref = val;
                    } else if (test_mode == NORM_SUM_DIV2_ONLY) {
                        ref = val * 0.5;
                    }

                    double act = (double)dst[block_idx][i];
                    double err = std::abs(ref - act);
                    if (err > max_error) max_error = err;
                    total_sq_error += err * err;
                    count++;
                }
            }
        }

        double mse = total_sq_error / count;
        std::cout << "Mode " << mode_names[m] << " Result: MSE=" << mse << ", MAE=" << max_error << std::endl;

        if (mse > 1e-5 || max_error > 1e-3) {
            std::cout << "Mode " << mode_names[m] << " FAILED" << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "ALL CSIM MODES PASSED" << std::endl;
    } else {
        std::cout << "CSIM FAILED" << std::endl;
    }
}

int main() {
    test_norm_sum();
    return 0;
}

#endif // __SYNTHESIS__
