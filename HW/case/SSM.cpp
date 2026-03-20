#include "../src/common.h"
#include "../src/ssm.h"
#include "../src/utils.h"
#include "../testbench/data_loader.hpp"
#include "../testbench/tb_utils.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>

using std::cout;
using std::endl;
using std::flush;
using std::string;

// ============================================================================
// Top-level function
// ============================================================================
void top(
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
    #pragma HLS interface ap_ctrl_chain port=return
    
    #pragma HLS interface m_axi port=u_src bundle=in_u offset=slave depth=10000 max_read_burst_length=256
    #pragma HLS interface m_axi port=delta_src bundle=in_delta offset=slave depth=10000 max_read_burst_length=256
    #pragma HLS interface m_axi port=z_silu_src bundle=in_z_silu offset=slave depth=10000 max_read_burst_length=256
    #pragma HLS interface m_axi port=B_src bundle=in_B offset=slave depth=10000 max_read_burst_length=256
    #pragma HLS interface m_axi port=C_src bundle=in_C offset=slave depth=10000 max_read_burst_length=256
    #pragma HLS interface m_axi port=out bundle=out_r offset=slave depth=10000 max_read_burst_length=256
    
    #pragma HLS interface m_axi port=A_base bundle=weights_A offset=slave depth=1000 max_read_burst_length=256
    #pragma HLS interface m_axi port=D_base bundle=weights_D offset=slave depth=100 max_read_burst_length=256
    
    #pragma HLS interface s_axilite port=scan_dim bundle=control
    #pragma HLS interface s_axilite port=inner_dim bundle=control
    #pragma HLS interface s_axilite port=num_patches bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    
    compute_ssm_output(out, u_src, delta_src, z_silu_src, A_base, B_src, C_src, D_base, scan_dim, inner_dim, num_patches);
}

// ============================================================================
// Testbench code (for C simulation only, not synthesized)
// ============================================================================

#ifndef __SYNTHESIS__

// Test parameters
constexpr int LAYER_IDX = 0;

// Input/Output arrays
static patch_blocks_inner_t_fp REF_U;
static patch_blocks_inner_t_fp REF_DELTA;
static patch_blocks_inner_t_fp REF_Z_SILU;
static patch_blocks_state_t_fp REF_B;
static patch_blocks_state_t_fp REF_C;
static float REF_OUTPUT[NUM_PATCHES][NUM_FEATURE_BLOCKS_INNER][FEATURE_BLOCK_SIZE];
static scan_out_block_t DUT_OUTPUT[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];

void test_ssm() {
    cout << "Testing SSM module..." << endl;

    cout << "Loading input data... " << flush;
    string u_file = string(get_project_root()) + "/data/ssm_float32/layers." + std::to_string(LAYER_IDX) + ".mixer.scan.u.float32.bin";
    string delta_file = string(get_project_root()) + "/data/ssm_float32/layers." + std::to_string(LAYER_IDX) + ".mixer.scan.delta.float32.bin";
    string z_silu_file = string(get_project_root()) + "/data/ssm_float32/layers." + std::to_string(LAYER_IDX) + ".mixer.scan.z_silu.float32.bin";
    string B_file = string(get_project_root()) + "/data/ssm_float32/layers." + std::to_string(LAYER_IDX) + ".mixer.scan.B.float32.bin";
    string C_file = string(get_project_root()) + "/data/ssm_float32/layers." + std::to_string(LAYER_IDX) + ".mixer.scan.C.float32.bin";
    
    if (!load_data(u_file, REF_U)) {
        cout << "Failed to load u from " << u_file << endl;
        return;
    }
    if (!load_data(delta_file, REF_DELTA)) {
        cout << "Failed to load delta from " << delta_file << endl;
        return;
    }
    if (!load_data(z_silu_file, REF_Z_SILU)) {
        cout << "Failed to load z_silu from " << z_silu_file << endl;
        return;
    }
    if (!load_data(B_file, REF_B)) {
        cout << "Failed to load B from " << B_file << endl;
        return;
    }
    if (!load_data(C_file, REF_C)) {
        cout << "Failed to load C from " << C_file << endl;
        return;
    }

    static fm_block_t u_data[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
    static fm_block_t delta_data[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
    static fm_block_t z_silu_data[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
    static fm_block_t B_data[NUM_PATCHES * NUM_FEATURE_BLOCKS_STATE];
    static fm_block_t C_data[NUM_PATCHES * NUM_FEATURE_BLOCKS_STATE];
    
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_INNER; ++b) {
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                u_data[p * NUM_FEATURE_BLOCKS_INNER + b][o] = (fm_t)REF_U[p][b][o];
                delta_data[p * NUM_FEATURE_BLOCKS_INNER + b][o] = (fm_t)REF_DELTA[p][b][o];
                z_silu_data[p * NUM_FEATURE_BLOCKS_INNER + b][o] = (fm_t)REF_Z_SILU[p][b][o];
            }
        }
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_STATE; ++b) {
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                B_data[p * NUM_FEATURE_BLOCKS_STATE + b][o] = (fm_t)REF_B[p][b][o];
                C_data[p * NUM_FEATURE_BLOCKS_STATE + b][o] = (fm_t)REF_C[p][b][o];
            }
        }
    }
    cout << "done" << endl;

    cout << "Loading A and D... " << flush;
    string A_file = string(get_project_root()) + "/data/ssm_float32/layers." + std::to_string(LAYER_IDX) + ".mixer.scan.A.float32.bin";
    string D_file = string(get_project_root()) + "/data/ssm_float32/layers." + std::to_string(LAYER_IDX) + ".mixer.scan.D.float32.bin";

    static fm_t A_array[SCAN_DIM];
    if (!load_data(A_file, A_array)) {
        cout << "Failed to load A from " << A_file << endl;
        return;
    }

    constexpr unsigned int A_BASE_SIZE = ceildiv(SCAN_DIM, 8U);
    static wt_wide_t A_base[A_BASE_SIZE];
    for (unsigned int i = 0; i < A_BASE_SIZE; ++i) {
        wt_wide_t word = 0;
        for (unsigned int j = 0; j < 8; ++j) {
            unsigned int idx = i * 8 + j;
            if (idx < SCAN_DIM) {
                ap_uint<32> elem_bits = A_array[idx].range(31, 0);
                word.range(j * 32 + 31, j * 32) = elem_bits;
            }
        }
        A_base[i] = word;
    }

    static fm_t D_array[D_INNER];
    if (!load_data(D_file, D_array)) {
        cout << "Failed to load D from " << D_file << endl;
        return;
    }

    constexpr unsigned int D_BASE_SIZE = ceildiv(D_INNER, 8U);
    static wt_wide_t D_base[D_BASE_SIZE];
    for (unsigned int i = 0; i < D_BASE_SIZE; ++i) {
        wt_wide_t word = 0;
        for (unsigned int j = 0; j < 8; ++j) {
            unsigned int idx = i * 8 + j;
            if (idx < D_INNER) {
                ap_uint<32> elem_bits = D_array[idx].range(31, 0);
                word.range(j * 32 + 31, j * 32) = elem_bits;
            }
        }
        D_base[i] = word;
    }
    cout << "done" << endl;

    cout << "Running SSM operation... " << flush;
    top(
        DUT_OUTPUT,
        u_data,
        delta_data,
        z_silu_data,
        A_base,
        B_data,
        C_data,
        D_base,
        SCAN_DIM,
        D_INNER,
        NUM_PATCHES
    );
    cout << "done" << endl;

    cout << "Loading reference output... " << flush;
    string ref_output_file = string(get_project_root()) + "/data/ssm_float32/layers." + std::to_string(LAYER_IDX) + ".mixer.scan.output.float32.bin";
    ifstream ref_ifs(ref_output_file, std::ios::binary);
    if (!ref_ifs) {
        cout << "Failed to load reference output from " << ref_output_file << endl;
        return;
    }
    ref_ifs.read(reinterpret_cast<char*>(REF_OUTPUT), sizeof(REF_OUTPUT));
    if (!ref_ifs) {
        cout << "Failed to read reference output" << endl;
        return;
    }
    cout << "done" << endl;

    cout << "Comparing results..." << endl;
    float mse = 0.0;
    float mae = 0.0;
    float max_err = 0.0;
    unsigned int total_elems = NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER * FEATURE_BLOCK_SIZE;
    
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_INNER; ++b) {
            unsigned int flat_idx = p * NUM_FEATURE_BLOCKS_INNER + b;
            for (unsigned int o = 0; o < FEATURE_BLOCK_SIZE; ++o) {
                float ref_val = REF_OUTPUT[p][b][o];
                float dut_val = (float)DUT_OUTPUT[flat_idx][o];
                float err = ref_val - dut_val;
                float abs_err = (err < 0) ? -err : err;
                mse += err * err;
                mae += abs_err;
                if (abs_err > max_err) max_err = abs_err;
            }
        }
    }
    
    mse /= total_elems;
    mae /= total_elems;
    
    cout << "SSM Output Comparison:" << endl;
    cout << "  MSE: " << mse << endl;
    cout << "  MAE: " << mae << endl;
    cout << "  Max Error: " << max_err << endl;
    
    if (mse < 0.01 && mae < 0.05) {
        cout << "SSM test PASSED" << endl;
    } else {
        cout << "SSM test FAILED" << endl;
    }
}

int main() {
    test_ssm();
    cout << "SSM test completed" << endl;
    return 0;
}

#endif // __SYNTHESIS__
