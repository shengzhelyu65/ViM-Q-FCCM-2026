#include "../src/common.h"
#include "../src/linear_block.h"
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
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_wide_t weights_base[],
    const wt_wide_t weight_scales_base[],
    const wt_wide_t bias_base[],
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int num_patches,
    int flags
)
{
    #pragma HLS interface ap_ctrl_chain port=return
    
    // AXI master interfaces
    #pragma HLS interface m_axi port=src bundle=in offset=slave depth=9456 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS interface m_axi port=dst bundle=out offset=slave depth=9456 max_read_burst_length=256 max_write_burst_length=256
    
    #pragma HLS interface m_axi port=weights_base bundle=weights offset=slave depth=1152 max_read_burst_length=256
    #pragma HLS interface m_axi port=weight_scales_base bundle=weights offset=slave depth=2304 max_read_burst_length=256
    #pragma HLS interface m_axi port=bias_base bundle=weights offset=slave depth=384 max_read_burst_length=256
    
    // AXI-Lite control interface for scalars
    #pragma HLS interface s_axilite port=out_dim bundle=control
    #pragma HLS interface s_axilite port=in_dim bundle=control
    #pragma HLS interface s_axilite port=num_patches bundle=control
    #pragma HLS interface s_axilite port=flags bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    
    compute_linear_block(dst, src, weights_base, weight_scales_base, bias_base,
                         out_dim, in_dim, num_patches, flags);
}

// ============================================================================
// Testbench code (for C simulation only, not synthesized)
// ============================================================================

#ifndef __SYNTHESIS__

// Test parameters
constexpr int IN_DIM = D_MODEL;
constexpr int OUT_DIM = D_INNER;
constexpr int LAYER_IDX = 0;

// Input/Output arrays
static fm_block_t REF_INPUT[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
static fm_block_t DUT_OUTPUT[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
static patch_blocks_inner_t_fp REF_OUTPUT;

// Helper function to pack float32 scales into wt_wide_t format
void pack_weight_scale_to_wt_wide(const float* src, wt_wide_t* dst, unsigned int count) {
    constexpr unsigned int FLOATS_PER_WORD = 8;
    unsigned int num_words = ceildiv(count, FLOATS_PER_WORD);
    
    for (unsigned int w = 0; w < num_words; w++) {
        wt_wide_t word = 0;
        for (unsigned int i = 0; i < FLOATS_PER_WORD; i++) {
            unsigned int idx = w * FLOATS_PER_WORD + i;
            ap_uint<32> bits = 0;
            if (idx < count) {
                wt_linear_ws_t val = (wt_linear_ws_t)src[idx];
                bits = val.range(31, 0);
            }
            word.range(i * 32 + 31, i * 32) = bits;
        }
        dst[w] = word;
    }
}

// Helper function to pack float32 bias values into wt_wide_t format
void pack_bias_to_wt_wide(const float* src, wt_wide_t* dst, unsigned int count) {
    constexpr unsigned int BIAS_PER_WORD = 8;
    unsigned int num_words = ceildiv(count, BIAS_PER_WORD);
    
    for (unsigned int w = 0; w < num_words; w++) {
        wt_wide_t word = 0;
        for (unsigned int i = 0; i < BIAS_PER_WORD; i++) {
            unsigned int idx = w * BIAS_PER_WORD + i;
            ap_uint<32> bits = 0;
            if (idx < count) {
                wt_linear_bias_t val = (wt_linear_bias_t)src[idx];
                bits = val.range(31, 0);
            }
            word.range(i * 32 + 31, i * 32) = bits;
        }
        dst[w] = word;
    }
}

bool load_and_pack_weights_4bit(
    const string& weight_file,
    wt_wide_t* weights_packed,
    unsigned int total_elements,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int offset_elements = 0  // Offset to skip (e.g., for X projection, skip Z)
) {
    (void)out_dim;
    (void)in_dim;

    std::ifstream ifs(weight_file, std::ios::binary);
    if (!ifs) {
        cout << "Failed to open " << weight_file << endl;
        return false;
    }

    unsigned int start_byte = offset_elements / 2;
    unsigned int start_nibble = offset_elements % 2; // 0: upper nibble, 1: lower nibble
    ifs.seekg(start_byte, std::ios::beg);
    if (!ifs) {
        cout << "Failed to seek in " << weight_file << endl;
        return false;
    }

    unsigned int num_words = ceildiv(total_elements, WEIGHTS_PER_WORD);

    unsigned char cur_byte = 0;
    bool have_byte = false;
    unsigned int nibble = start_nibble;

    for (unsigned int w = 0; w < num_words; w++) {
        wt_wide_t word = 0;

        for (unsigned int i = 0; i < WEIGHTS_PER_WORD; i++) {
            unsigned int idx = w * WEIGHTS_PER_WORD + i;

            ap_uint<4> weight_4bit = 0;
            if (idx < total_elements) {
                if (!have_byte) {
                    ifs.read(reinterpret_cast<char*>(&cur_byte), 1);
                    if (!ifs) {
                        cout << "Error reading weights from " << weight_file << endl;
                        return false;
                    }
                    have_byte = true;
                }

                weight_4bit = (nibble == 0) ? ap_uint<4>((cur_byte >> 4) & 0x0F) : ap_uint<4>(cur_byte & 0x0F);
                if (nibble == 0) {
                    nibble = 1;
                } else {
                    nibble = 0;
                    have_byte = false;
                }
            }

            // Pack into word
            unsigned int bit_offset = i * 4;
            word.range(bit_offset + 3, bit_offset) = weight_4bit;
        }

        weights_packed[w] = word;
    }
    
    cout << "Loaded and packed " << total_elements << " weights into " << num_words << " words" << endl;
    
    return true;
}

void test_linear_in_proj() {
    cout << "Testing LINEAR_BLOCK (in_proj) module..." << endl;
    const char* project_root = get_project_root();

    // Load input data
    cout << "Loading input data... " << flush;
    patch_blocks_t_fp input_fp;
    string input_file = string(project_root) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.in_proj.input.float32.bin";
    if (!load_data(input_file, input_fp)) {
        cout << "Failed to load input from " << input_file << endl;
        return;
    }
    
    for (unsigned int i = 0; i < NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER; ++i) {
        for (unsigned int j = 0; j < FEATURE_BLOCK_SIZE; ++j) {
            REF_INPUT[i][j] = (fm_t)0.0;
        }
    }

    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS; ++b) {
            for (unsigned int i = 0; i < FEATURE_BLOCK_SIZE; ++i) {
                REF_INPUT[p * NUM_FEATURE_BLOCKS + b][i] = (fm_t)input_fp[p][b][i];
            }
        }
    }
    cout << "done" << endl;

    // Load and pack weights
    cout << "Loading and packing weights... " << flush;
    string weight_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.in_proj.weight.x.int4.bin";
    
    // X projection only: D_INNER * D_MODEL = 384 * 192 = 73,728
    unsigned int total_elems = D_INNER * D_MODEL;  // Elements we want (X only)
    static wt_wide_t weights_packed[ceildiv(D_INNER * D_MODEL, WEIGHTS_PER_WORD)];
    
    // Load X projection (no offset needed since file contains only X)
    if (!load_and_pack_weights_4bit(weight_file, weights_packed, total_elems, OUT_DIM, IN_DIM, 0)) {
        cout << "Failed to load and pack weights" << endl;
        return;
    }
    cout << "done" << endl;

    // Load weight scales for X projection (per-block format: [out_dim][num_blocks])
    cout << "Loading weight scales... " << flush;
    string scale_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.in_proj.weight_scale.x.float32.bin";
    
    unsigned int num_in_blocks = ceildiv((unsigned int)IN_DIM, (unsigned int)WEIGHT_BLOCK_SIZE);  // 192/32 = 6
    unsigned int scales_per_proj = D_INNER * num_in_blocks;  // 384 * 6 = 2304
    
    // Allocate packed scales in wt_wide_t format (8 float32 per 256-bit word)
    constexpr unsigned int SCALES_PER_WORD_TEST = 8;
    static wt_wide_t weight_scales_packed[ceildiv(D_INNER * ceildiv((unsigned int)D_MODEL, (unsigned int)WEIGHT_BLOCK_SIZE), SCALES_PER_WORD_TEST)];

    // Read scales directly
    static float temp_scales[D_INNER * ceildiv((unsigned int)D_MODEL, (unsigned int)WEIGHT_BLOCK_SIZE)];
    if (!load_data(scale_file, temp_scales)) {
        cout << "Failed to load weight scales from " << scale_file << endl;
        return;
    }
    
    // Pack scales into wt_wide_t words
    for (unsigned int w = 0; w < ceildiv(scales_per_proj, SCALES_PER_WORD_TEST); w++) {
        wt_wide_t word = 0;
        for (unsigned int i = 0; i < SCALES_PER_WORD_TEST; i++) {
            unsigned int idx = w * SCALES_PER_WORD_TEST + i;
            ap_uint<32> scale_bits = 0;
            if (idx < scales_per_proj) {
                wt_linear_ws_t scale_val = (wt_linear_ws_t)temp_scales[idx];
                scale_bits = scale_val.range(31, 0);
            }
            word.range(i * 32 + 31, i * 32) = scale_bits;
        }
        weight_scales_packed[w] = word;
    }
    cout << "done" << endl;
    
    // Initialize bias to zero (in_proj doesn't use bias)  
    static wt_wide_t bias_packed[ceildiv((unsigned int)D_INNER, SCALES_PER_WORD_TEST)];
    for (unsigned int i = 0; i < ceildiv((unsigned int)D_INNER, SCALES_PER_WORD_TEST); ++i) {
        bias_packed[i] = 0;
    }
    
    cout << "Running LINEAR_BLOCK operation... " << flush;
    int flags = 0;  // No bias, no SiLU for X projection
    top(
        DUT_OUTPUT,
        REF_INPUT,
        weights_packed,
        weight_scales_packed,
        bias_packed,
        OUT_DIM,
        IN_DIM,
        NUM_PATCHES,
        flags
    );
    cout << "done" << endl;

    // Load reference output
    cout << "Loading reference output... " << flush;
    string ref_file = string(project_root) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.in_proj.output.x.float32.bin";
    if (!load_data(ref_file, REF_OUTPUT)) {
        cout << "Failed to load reference output from " << ref_file << endl;
        return;
    }
    cout << "done" << endl;
    
    // Compare results
    cout << "Comparing results..." << endl;
    perform_complete_comparison(
        reinterpret_cast<const patch_blocks_inner_t&>(DUT_OUTPUT),
        REF_OUTPUT,
        "LINEAR_BLOCK (in_proj) output",
        "In Projection Output",
        NUM_PATCHES,
        D_INNER,
        CLS_TOKEN_IDX
    );
}

void test_linear_delta() {
    cout << "\nTesting LINEAR_BLOCK for delta path (smooth -> x_proj_dt -> dt_proj)..." << endl;
    const char* project_root = get_project_root();

    // 1. Load conv1d.output
    cout << "Loading conv1d output data... " << flush;
    patch_blocks_inner_t_fp conv1d_out_fp;
    string conv1d_file = string(project_root) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.x_proj.input.float32.bin";
    if (!load_data(conv1d_file, conv1d_out_fp)) {
        cout << "Failed to load conv1d output from " << conv1d_file << endl;
        return;
    }
    cout << "done" << endl;

    // 2. Load smooth scales
    cout << "Loading smooth scales... " << flush;
    string smooth_scale_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.x_proj.smooth_scales.float32.bin";
    static float smooth_scales[D_INNER];
    if (!load_data(smooth_scale_file, smooth_scales)) {
        cout << "Failed to load smooth scales from " << smooth_scale_file << endl;
        return;
    }
    cout << "done" << endl;

    // 3. Perform smooth in CPU: x_proj_input = conv1d_output * smooth_scales
    static fm_block_t XPROJ_DT_INPUT[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER];
    for (unsigned int p = 0; p < NUM_PATCHES; ++p) {
        for (unsigned int b = 0; b < NUM_FEATURE_BLOCKS_INNER; ++b) {
            for (unsigned int i = 0; i < FEATURE_BLOCK_SIZE; ++i) {
                unsigned int dim = b * FEATURE_BLOCK_SIZE + i;
                if (dim < D_INNER) {
                    XPROJ_DT_INPUT[p * NUM_FEATURE_BLOCKS_INNER + b][i] = (fm_t)(conv1d_out_fp[p][b][i] * smooth_scales[dim]);
                } else {
                    XPROJ_DT_INPUT[p * NUM_FEATURE_BLOCKS_INNER + b][i] = (fm_t)0.0;
                }
            }
        }
    }

    // 4. Run x_proj_dt (Linear block)
    // out_dim = 16, in_dim = 384
    constexpr unsigned int XPROJ_DT_OUT_DIM = DT_RANK; // 16
    constexpr unsigned int XPROJ_DT_IN_DIM = D_INNER;  // 384
    constexpr unsigned int XPROJ_DT_ELEMS = XPROJ_DT_OUT_DIM * XPROJ_DT_IN_DIM; // 16 * 384 = 6144

    cout << "Loading and packing x_proj_dt weights... " << flush;
    string weight_dt_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.x_proj.weight.dt.int4.bin";
    static wt_wide_t weights_packed_dt[ceildiv(XPROJ_DT_ELEMS, (unsigned int)WEIGHTS_PER_WORD)];
    for (unsigned int i = 0; i < ceildiv(XPROJ_DT_ELEMS, (unsigned int)WEIGHTS_PER_WORD); i++) weights_packed_dt[i] = 0;
    if (!load_and_pack_weights_4bit(weight_dt_file, weights_packed_dt, XPROJ_DT_ELEMS, XPROJ_DT_OUT_DIM, XPROJ_DT_IN_DIM, 0)) {
        cout << "Failed to load and pack x_proj_dt weights" << endl;
        return;
    }
    cout << "done" << endl;

    cout << "Loading x_proj_dt weight scales... " << flush;
    string scale_dt_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.x_proj.weight_scale.dt.float32.bin";
    constexpr unsigned int num_blocks_dt = ceildiv(XPROJ_DT_IN_DIM, (unsigned int)WEIGHT_BLOCK_SIZE); // 384/32 = 12
    constexpr unsigned int scales_dt_count = XPROJ_DT_OUT_DIM * num_blocks_dt; // 16 * 12 = 192
    static float temp_scales_dt[scales_dt_count];
    if (!load_data(scale_dt_file, temp_scales_dt)) {
        cout << "Failed to load x_proj_dt weight scales from " << scale_dt_file << endl;
        return;
    }
    static wt_wide_t weight_scales_packed_dt[ceildiv(scales_dt_count, 8u)];
    pack_weight_scale_to_wt_wide(temp_scales_dt, weight_scales_packed_dt, scales_dt_count);
    cout << "done" << endl;

    static wt_wide_t bias_packed_dummy_dt[ceildiv((unsigned int)D_INNER, 8u)];
    static float zeros_dt[D_INNER] = {0};
    pack_weight_scale_to_wt_wide(zeros_dt, bias_packed_dummy_dt, D_INNER);

    cout << "Running LINEAR_BLOCK for x_proj_dt... " << flush;
    static fm_block_t XPROJ_DT_OUT[NUM_PATCHES * ceildiv(DT_RANK, FEATURE_BLOCK_SIZE)];
    top(
        XPROJ_DT_OUT,
        XPROJ_DT_INPUT,
        weights_packed_dt,
        weight_scales_packed_dt,
        bias_packed_dummy_dt,
        XPROJ_DT_OUT_DIM,
        XPROJ_DT_IN_DIM,
        NUM_PATCHES,
        0
    );
    cout << "done" << endl;

    // 5. Run dt_proj (Linear block)
    // in_dim = 16, out_dim = 384
    constexpr unsigned int DTPROJ_OUT_DIM = D_INNER; // 384
    constexpr unsigned int DTPROJ_IN_DIM = DT_RANK;  // 16
    constexpr unsigned int DTPROJ_ELEMS = DTPROJ_OUT_DIM * DTPROJ_IN_DIM; // 384 * 16 = 6144

    cout << "Loading and packing dt_proj weights... " << flush;
    string weight_dp_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.dt_proj.weight.int4.bin";
    static wt_wide_t weights_packed_dp[ceildiv(DTPROJ_ELEMS, (unsigned int)WEIGHTS_PER_WORD)];
    for (unsigned int i = 0; i < ceildiv(DTPROJ_ELEMS, (unsigned int)WEIGHTS_PER_WORD); i++) weights_packed_dp[i] = 0;
    if (!load_and_pack_weights_4bit(weight_dp_file, weights_packed_dp, DTPROJ_ELEMS, DTPROJ_OUT_DIM, DTPROJ_IN_DIM, 0)) {
        cout << "Failed to load and pack dt_proj weights" << endl;
        return;
    }
    cout << "done" << endl;

    cout << "Loading dt_proj weight scales... " << flush;
    string scale_dp_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.dt_proj.weight_scale.float32.bin";
    constexpr unsigned int num_blocks_dp = ceildiv(DTPROJ_IN_DIM, (unsigned int)WEIGHT_BLOCK_SIZE); // 16/32 = 1
    constexpr unsigned int scales_dp_count = DTPROJ_OUT_DIM * num_blocks_dp; // 384 * 1 = 384
    static float temp_scales_dp[scales_dp_count];
    if (!load_data(scale_dp_file, temp_scales_dp)) {
        cout << "Failed to load dt_proj weight scales from " << scale_dp_file << endl;
        return;
    }
    static wt_wide_t weight_scales_packed_dp[ceildiv(scales_dp_count, 8u)];
    pack_weight_scale_to_wt_wide(temp_scales_dp, weight_scales_packed_dp, scales_dp_count);
    cout << "done" << endl;

    cout << "Loading dt_proj bias... " << flush;
    string bias_dp_file = string(project_root) + "/data/bin_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.dt_proj.bias.float32.bin";
    static float temp_bias_dp[DTPROJ_OUT_DIM];
    if (!load_data(bias_dp_file, temp_bias_dp)) {
        cout << "Failed to load dt_proj bias from " << bias_dp_file << endl;
        return;
    }
    static wt_wide_t bias_packed_dp[ceildiv(DTPROJ_OUT_DIM, 8u)];
    pack_bias_to_wt_wide(temp_bias_dp, bias_packed_dp, DTPROJ_OUT_DIM);
    cout << "done" << endl;

    cout << "Running LINEAR_BLOCK for dt_proj... " << flush;
    static fm_block_t DT_PROJ_OUT[NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER]; // 197 * 24
    top(
        DT_PROJ_OUT,
        XPROJ_DT_OUT,
        weights_packed_dp,
        weight_scales_packed_dp,
        bias_packed_dp,
        DTPROJ_OUT_DIM,
        DTPROJ_IN_DIM,
        NUM_PATCHES,
        FLAG_BIAS | FLAG_SOFTPLUS
    );
    cout << "done" << endl;

    // 6. Compare results
    cout << "Loading reference delta output... " << flush;
    patch_blocks_inner_t_fp ref_delta_fp;
    string ref_delta_file = string(project_root) + "/data/ref_float32_block/layers." + std::to_string(LAYER_IDX) + ".mixer.delta.float32.bin";
    if (!load_data(ref_delta_file, ref_delta_fp)) {
        cout << "Failed to load reference delta from " << ref_delta_file << endl;
        return;
    }
    cout << "done" << endl;

    perform_complete_comparison(
        reinterpret_cast<const patch_blocks_inner_t&>(DT_PROJ_OUT),
        ref_delta_fp,
        "LINEAR_BLOCK (delta path) output",
        "Final Delta Output",
        NUM_PATCHES,
        D_INNER,
        CLS_TOKEN_IDX
    );
}

int main() {
    test_linear_in_proj();
    test_linear_delta();
    cout << "\nLINEAR_BLOCK tests completed" << endl;
    return 0;
}

#endif // __SYNTHESIS__
