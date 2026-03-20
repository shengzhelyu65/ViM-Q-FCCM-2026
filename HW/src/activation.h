#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include "common.h"
#include "silu_table.h"
#include "softplus_table.h"
#include <hls_math.h>
#include <hls_vector.h>

// Scalar SiLU
inline fm_t silu(fm_t x)
{
    #pragma HLS inline
    #pragma HLS bind_storage variable=SILU_DELTA_TABLE type=rom_np impl=bram

    fm_t relu = ap_fixed_relu(x);
    fm_t x_abs = hls::signbit(x) ? fm_t(-x) : x;

    if (x_abs >= SILU_DELTA_TABLE_MAX) return relu;

    auto index_exact = x_abs * SILU_INV_DELTA_TABLE_STEP;
    silu_delta_table_index_t index = (unsigned int)index_exact;

    fm_frac_t a = SILU_DELTA_TABLE[index];
    fm_frac_t b = SILU_DELTA_TABLE[index + 1];
    auto t = index_exact - (fm_t)index;

    fm_frac_t silu_delta = a + t * (b - a);
    return relu - (fm_t)silu_delta;
}

// Vector SiLU
template<size_t N>
inline hls::vector<fm_t, N> silu(hls::vector<fm_t, N> x)
{
    hls::vector<fm_t, N> result;
    for (size_t i = 0; i < N; i++)
    {
        #pragma HLS unroll
        result[i] = silu(x[i]);
    }
    return result;
}

// Scalar Softplus
inline fm_t softplus(fm_t x)
{
    #pragma HLS inline
    #pragma HLS bind_storage variable=SOFTPLUS_DELTA_TABLE type=rom_np impl=bram

    fm_t relu = ap_fixed_relu(x);
    fm_t x_abs = hls::signbit(x) ? fm_t(-x) : x;

    if (x_abs >= SOFTPLUS_DELTA_TABLE_MAX) return relu;

    auto index_exact = x_abs * SOFTPLUS_INV_DELTA_TABLE_STEP;
    softplus_delta_table_index_t index = (unsigned int)index_exact;

    fm_frac_t a = SOFTPLUS_DELTA_TABLE[index];
    fm_frac_t b = SOFTPLUS_DELTA_TABLE[index + 1];
    auto t = index_exact - (fm_t)index;

    fm_frac_t softplus_delta = a + t * (b - a);
    return relu + (fm_t)softplus_delta;
}

// Vector Softplus
template<size_t N>
inline hls::vector<fm_t, N> softplus(hls::vector<fm_t, N> x)
{
    hls::vector<fm_t, N> result;
    for (size_t i = 0; i < N; i++)
    {
        #pragma HLS unroll
        result[i] = softplus(x[i]);
    }
    return result;
}

#endif // __ACTIVATION_H__

