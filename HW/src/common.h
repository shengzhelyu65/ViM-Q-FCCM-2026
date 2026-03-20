#ifndef __INT_COMMON_H__
#define __INT_COMMON_H__

#include <fstream>
#include <vector> 
#include <string> 
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <iostream>
#include <cstdint>

// HLS library
#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <hls_burst_maxi.h>

using namespace std;

// ============================================================================
// Model Parameters
// ============================================================================
constexpr unsigned int INPUT_CHANNELS = 3;
constexpr unsigned int INPUT_WIDTH = 224;
constexpr unsigned int INPUT_HEIGHT = 224;
constexpr unsigned int PATCH_WIDTH = 16;
constexpr unsigned int PATCH_HEIGHT = 16;

constexpr unsigned int CLS_TOKEN_IDX = (INPUT_WIDTH / PATCH_WIDTH) * (INPUT_HEIGHT / PATCH_HEIGHT) / 2;
constexpr unsigned int NUM_PATCHES = (INPUT_WIDTH / PATCH_WIDTH) * (INPUT_HEIGHT / PATCH_HEIGHT) + 1;
constexpr unsigned int CONV_KERNEL_SIZE = 4;

#ifndef __SYNTHESIS__
constexpr unsigned int NUM_LAYERS = 1;
#else
constexpr unsigned int NUM_LAYERS = 24;
#endif

constexpr unsigned int D_MODEL = 192;
constexpr unsigned int D_INNER = 2 * D_MODEL;
constexpr unsigned int D_STATE = 16;
constexpr unsigned int DT_RANK = 16;
constexpr unsigned int SCAN_DIM = D_INNER * D_STATE;
constexpr unsigned int PADDED_NUM_CLASSES = 1008;

// Constants for BRAM sizing
constexpr unsigned int MAX_D_MODEL = 384;
constexpr unsigned int MAX_D_INNER = 2 * MAX_D_MODEL;
constexpr unsigned int MAX_SCAN_DIM = MAX_D_INNER * D_STATE;

constexpr unsigned int MAX_LINEAR_IN_DIM = D_INNER;
constexpr unsigned int MAX_LINEAR_OUT_DIM = PADDED_NUM_CLASSES;

// ============================================================================
// Data Types
// ============================================================================
typedef ap_fixed<32, 10> pixel_t;
typedef pixel_t image_t[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];

typedef ap_fixed<32, 14> fm_t;
typedef ap_fixed<32, 14> gate_t;
typedef ap_fixed<32, 14> token_t;
typedef ap_fixed<32, 14> scan_t;

// Patch embedding
typedef ap_fixed<16, 1> wt_patch_embed_t;
typedef ap_fixed<16, 4> wt_patch_bias_t;

// RMS Norm
typedef ap_fixed<16, 1> wt_norm_t;

// Linear Layer
typedef ap_uint<4> wt_linear_t;
typedef ap_fixed<32, 14> wt_linear_bias_t;
typedef ap_ufixed<32, 6> wt_linear_ws_t;
typedef ap_ufixed<32, 14> wt_linear_as_t;
typedef ap_ufixed<32, 6> wt_linear_ss_t;

typedef bool wt_conv_sign_t;
typedef ap_uint<4> wt_conv_mag_t;
typedef ap_fixed<32, 14> wt_conv_bias_t;
typedef ap_ufixed<32, 6> wt_conv_ws_t;
typedef ap_ufixed<32, 14> wt_conv_as_t;

// ============================================================================
// Utility Functions (from util.hpp)
// ============================================================================
#define _LABEL_FOR_EACH_HELPER(line, var) _ln ## line ## _for_each_ ## var
#define _LABEL_FOR_EACH(line, var) _LABEL_FOR_EACH_HELPER(line, var)
#define FOR_EACH(var, limit) _LABEL_FOR_EACH(__LINE__, var): for (unsigned int var = 0; var < limit; var++)

#define _LABEL_FOR_BLOCK_HELPER(line, var) _ln ## line ## _for_block_ ## var
#define _LABEL_FOR_BLOCK(line, var) _LABEL_FOR_BLOCK_HELPER(line, var)
#define FOR_BLOCK(var, limit, block_size) \
    constexpr unsigned int var##_step = (block_size); \
    constexpr unsigned int var##_limit = (limit); \
    constexpr unsigned int var##_iters = ((limit) + (block_size) - 1) / (block_size); \
    _LABEL_FOR_BLOCK(__LINE__, var): for ( \
        unsigned int var##_base = 0, var##_block = 0; \
        var##_base < var##_limit; \
        var##_base += var##_step, var##_block++ \
    )

#define _LABEL_FOR_OFFSET_HELPER(line, var) _ln ## line ## _for_offset_ ## var
#define _LABEL_FOR_OFFSET(line, var) _LABEL_FOR_OFFSET_HELPER(line, var)
#define FOR_OFFSET(var) \
    _LABEL_FOR_OFFSET(__LINE__, var): for ( \
        unsigned int var##_offset = 0, var = var##_base; \
        var##_offset < var##_step; \
        var##_offset++, var++ \
    ) \
    if (var##_limit % var##_step == 0 || var < var##_limit)

#define _LABEL_FOR_OFFSET_NOCHK_HELPER(line, var) _ln ## line ## _for_offset_ ## var
#define _LABEL_FOR_OFFSET_NOCHK(line, var) _LABEL_FOR_OFFSET_NOCHK_HELPER(line, var)
#define FOR_OFFSET_NOCHK(var) \
    static_assert(var##_limit % var##_step == 0, "Cannot use FOR_OFFSET_NOCHK; use FOR_OFFSET instead."); \
    _LABEL_FOR_OFFSET_NOCHK(__LINE__, var): for ( \
        unsigned int var##_offset = 0, var = var##_base; \
        var##_offset < var##_step; \
        var##_offset++, var++ \
    )

#define _LABEL_FOR_OFFSET_UNSAFE_HELPER(line, var) _ln ## line ## _for_offset_ ## var
#define _LABEL_FOR_OFFSET_UNSAFE(line, var) _LABEL_FOR_OFFSET_UNSAFE_HELPER(line, var)
#define FOR_OFFSET_UNSAFE(var) \
    _LABEL_FOR_OFFSET_UNSAFE(__LINE__, var): for ( \
        unsigned int var##_offset = 0, var = var##_base; \
        var##_offset < var##_step; \
        var##_offset++, var++ \
    )

template <typename T>
static constexpr T ceildiv(T dividend, T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
static constexpr T roundup(T dividend, T divisor)
{
    return ceildiv(dividend, divisor) * divisor;
}

template <typename T>
static constexpr T max(T a, T b)
{
    return (a > b) ? a : b;
}

template <typename T>
static constexpr T roundup_p2(T num)
{
    return (num == 0 || num == 1) ? 1 : 2 * roundup_p2((num + 1) / 2);
}

template <typename T>
static constexpr T bitcount(T num)
{
    return (num == 0 || num == 1) ? num : 1 + bitcount(num >> 1);
}

template <typename T>
static constexpr T ap_fixed_relu(T x)
{
    return hls::signbit(x) ? T(0) : x;
}

template <typename T>
static constexpr T ap_fixed_epsilon()
{
    return T(1.0 / (1 << (T::width - T::iwidth)));
}

template <typename T>
static constexpr T ap_fixed_min()
{
    return T(-(1 << (T::iwidth - 1)));
}

// ============================================================================
// Design Space Exploration (from dse.hpp)
// ============================================================================
constexpr unsigned int VECTOR_SIZE = 4;
constexpr unsigned int NUM_VECTORS = ceildiv(NUM_PATCHES, VECTOR_SIZE);
constexpr unsigned int NUM_PADDED_VECTORS = roundup_p2(NUM_VECTORS);
constexpr unsigned int NUM_SWEEP = bitcount(NUM_PADDED_VECTORS) - 1;

// ============================================================================
// Hardware Constants (from hardware.hpp)
// ============================================================================
constexpr unsigned int AXI_XFER_BIT_WIDTH = 256;
constexpr unsigned int Q_MAX = 127;
constexpr unsigned int Q_MIN = -128;
constexpr float Q_MAX_FLOAT = 0.0078740157480315f;
constexpr float Q_MIN_FLOAT = -0.0078125f;

constexpr unsigned int FEATURE_BLOCK_SIZE = 8;
constexpr unsigned int LINEAR_BLOCK_SIZE = 16;
constexpr unsigned int CONV_BLOCK_SIZE = 16;

typedef ap_uint<AXI_XFER_BIT_WIDTH> wt_wide_t;
constexpr unsigned int MAGS_PER_WORD = AXI_XFER_BIT_WIDTH / 4;
constexpr unsigned int SIGNS_PER_WORD = AXI_XFER_BIT_WIDTH / 1;

typedef hls::vector<fm_t, FEATURE_BLOCK_SIZE> fm_block_t;

constexpr unsigned int NUM_FEATURE_BLOCKS = ceildiv(D_MODEL, FEATURE_BLOCK_SIZE);
typedef fm_block_t fm_blocks_t[NUM_FEATURE_BLOCKS];
typedef fm_blocks_t patch_blocks_t[NUM_PATCHES];
typedef fm_blocks_t patch_cls_token_t[1];

constexpr unsigned int NUM_FEATURE_BLOCKS_INNER = ceildiv(D_INNER, FEATURE_BLOCK_SIZE);
typedef fm_block_t fm_blocks_inner_t[NUM_FEATURE_BLOCKS_INNER];
typedef fm_blocks_inner_t patch_blocks_inner_t[NUM_PATCHES];

constexpr unsigned int NUM_FEATURE_BLOCKS_STATE = ceildiv(D_STATE, FEATURE_BLOCK_SIZE);
typedef fm_block_t fm_blocks_state_t[NUM_FEATURE_BLOCKS_STATE];
typedef fm_blocks_state_t patch_blocks_state_t[NUM_PATCHES];

constexpr unsigned int NUM_FEATURE_BLOCKS_SCAN = ceildiv(SCAN_DIM, FEATURE_BLOCK_SIZE);

typedef hls::vector<scan_t, FEATURE_BLOCK_SIZE> scan_out_block_t;
typedef scan_out_block_t fm_blocks_scan_t[NUM_FEATURE_BLOCKS_SCAN];
typedef fm_blocks_scan_t patch_blocks_scan_t[NUM_PATCHES];

typedef scan_out_block_t fm_blocks_inner_scan_t[NUM_FEATURE_BLOCKS_INNER];
typedef fm_blocks_inner_scan_t patch_blocks_inner_scan_t[NUM_PATCHES];

typedef hls::vector<gate_t, FEATURE_BLOCK_SIZE> gate_block_t;
typedef gate_block_t gate_blocks_scan_t[NUM_FEATURE_BLOCKS_SCAN];
typedef gate_blocks_scan_t patch_blocks_gate_t[NUM_PATCHES];

typedef hls::vector<token_t, FEATURE_BLOCK_SIZE> token_block_t;
typedef token_block_t token_blocks_scan_t[NUM_FEATURE_BLOCKS_SCAN];
typedef token_blocks_scan_t patch_blocks_token_t[NUM_PATCHES];

typedef fm_block_t final_result_t[ceildiv(PADDED_NUM_CLASSES, FEATURE_BLOCK_SIZE)];

// ============================================================================
// System Constants
// ============================================================================
const unsigned int SYSTEM_WIDTH = 256;  // AXI_XFER_BIT_WIDTH
typedef ap_uint<SYSTEM_WIDTH> system_t;

// Helper function for log2 ceiling
constexpr int log2ce(int n){
    return (n <= 1) ? 0 : 1 + log2ce(n / 2);
}

// HLS comprehensive attributes
constexpr const int BRAM_STYLE = 0;
constexpr const int URAM_STYLE = 1;
constexpr const int LRAM_STYLE = 2;

#endif
