#include "../src/embed.h"
#include "../src/common.h"
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
    fm_block_t *out,
    const wt_wide_t *image,
    const wt_wide_t *weights,
    const wt_wide_t *bias,
    const wt_wide_t *pos_embed,
    const wt_wide_t *cls_token,
    const unsigned int dim,
    const unsigned int numPatches,
    const unsigned int imgHeight,
    const unsigned int imgWidth
) {
    #pragma HLS interface ap_ctrl_chain port=return

    #pragma HLS interface m_axi port=out bundle=out offset=slave depth=4728 num_write_outstanding=16 max_write_burst_length=32
    #pragma HLS interface m_axi port=image bundle=in offset=slave depth=18816 num_read_outstanding=16 max_read_burst_length=32
    #pragma HLS interface m_axi port=weights bundle=weights offset=slave depth=9216 num_read_outstanding=16 max_read_burst_length=32
    #pragma HLS interface m_axi port=bias bundle=weights offset=slave depth=24 num_read_outstanding=16 max_read_burst_length=32
    #pragma HLS interface m_axi port=pos_embed bundle=weights offset=slave depth=4728 num_read_outstanding=16 max_read_burst_length=32
    #pragma HLS interface m_axi port=cls_token bundle=weights offset=slave depth=24 num_read_outstanding=16 max_read_burst_length=32

    #pragma HLS interface s_axilite port=dim bundle=control
    #pragma HLS interface s_axilite port=numPatches bundle=control
    #pragma HLS interface s_axilite port=imgHeight bundle=control
    #pragma HLS interface s_axilite port=imgWidth bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    patch_embed_impl(out, image, weights, bias, pos_embed, cls_token, dim, numPatches, imgHeight, imgWidth);
}

// ============================================================================
// Testbench code (for C simulation only, not synthesized)
// ============================================================================

#ifndef __SYNTHESIS__

void test_embed() {
    cout << "Testing patch_embed..." << endl;

    // Data structures for testbench
    static pixel_t image[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
    static wt_patch_embed_t weights[D_MODEL][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH];
    static wt_patch_bias_t bias[D_MODEL];

    static fm_block_t pos_embed[NUM_PATCHES * NUM_FEATURE_BLOCKS];
    static fm_block_t cls_token[NUM_FEATURE_BLOCKS];
    static fm_block_t dut_output[NUM_PATCHES * NUM_FEATURE_BLOCKS];
    static float ref_output[NUM_PATCHES][NUM_FEATURE_BLOCKS][FEATURE_BLOCK_SIZE];

    // Load data
    const char* project_root = get_project_root();
    cout << "Loading image... " << flush;
    if (!load_data(string(project_root) + "/data/image_float32_block/image.float32.bin", image)) return;
    cout << "done" << endl;

    cout << "Loading weights... " << flush;
    if (!load_data(string(project_root) + "/data/bin_float32_block/patch_embed.proj.weight.float32.bin", weights)) return;
    cout << "done" << endl;

    cout << "Loading bias... " << flush;
    if (!load_data(string(project_root) + "/data/bin_float32_block/patch_embed.proj.bias.float32.bin", bias)) return;
    cout << "done" << endl;

    cout << "Loading pos_embed... " << flush;
    if (!load_data(string(project_root) + "/data/bin_float32_block/pos_embed.float32.bin", pos_embed)) return;
    cout << "done" << endl;

    cout << "Loading cls_token... " << flush;
    if (!load_data(string(project_root) + "/data/bin_float32_block/cls_token.float32.bin", cls_token)) return;
    cout << "done" << endl;

    cout << "Loading reference output... " << flush;
    if (!load_data(string(project_root) + "/data/ref_float32_block/layers.0.norm.input.float32.bin", ref_output)) return;
    cout << "done" << endl;

    // Run DUT
    cout << "Running DUT... " << flush;
    top(
        dut_output,
        reinterpret_cast<const wt_wide_t*>(image),
        reinterpret_cast<const wt_wide_t*>(weights),
        reinterpret_cast<const wt_wide_t*>(bias),
        reinterpret_cast<const wt_wide_t*>(pos_embed),
        reinterpret_cast<const wt_wide_t*>(cls_token),
        D_MODEL,
        NUM_PATCHES,
        INPUT_HEIGHT,
        INPUT_WIDTH
    );
    cout << "done" << endl;

    // Compare
    cout << "Comparing results..." << endl;
    perform_complete_comparison(
        reinterpret_cast<const patch_blocks_t&>(dut_output),
        ref_output,
        "patch_embed output",
        "Embedded Patches",
        NUM_PATCHES,
        D_MODEL,
        CLS_TOKEN_IDX
    );
}

int main() {
    test_embed();
    cout << "EMBED test completed" << endl;
    return 0;
}

#endif // __SYNTHESIS__
