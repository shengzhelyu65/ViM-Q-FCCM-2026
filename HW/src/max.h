#ifndef __MAX_H__
#define __MAX_H__

#include "common.h"

template<typename T>
T max_block(hls::vector<T, 16>& block) {
    #pragma HLS inline off

    T working_storage[8];
    #pragma HLS ARRAY_PARTITION variable=working_storage complete dim=1

    // Stage 1: Combined absolute value and first reduction
    first_stage_loop: for (size_t i = 0; i < 8; i++) {
        #pragma HLS UNROLL
        T val1 = block[2 * i];
        T val2 = block[2 * i + 1];
        T abs_val1 = (val1 < (T)0) ? (T)-val1 : val1;
        T abs_val2 = (val2 < (T)0) ? (T)-val2 : val2;
        working_storage[i] = (abs_val1 > abs_val2) ? abs_val1 : abs_val2;
    }

    // Subsequent reduction stages
    reduction_outer_loop: for (size_t current_active_elements = 8; current_active_elements > 1; current_active_elements /= 2) {
        size_t pairs_to_compare = current_active_elements / 2;
        reduction_inner_loop_opt: for (size_t i = 0; i < pairs_to_compare; i++) {
            #pragma HLS UNROLL
            working_storage[i] = (working_storage[2 * i] > working_storage[2 * i + 1]) ? working_storage[2 * i] : working_storage[2 * i + 1];
        }
    }
    return working_storage[0];
}

template<typename T>
T max_block(hls::vector<T, 8>& block) {
    #pragma HLS inline off

    T working_storage[4];
    #pragma HLS ARRAY_PARTITION variable=working_storage complete dim=1

    // Stage 1: Process 8 elements -> 4 elements (absolute value and first reduction)
    first_stage_loop: for (size_t i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        T val1 = block[2 * i];
        T val2 = block[2 * i + 1];
        T abs_val1 = (val1 < (T)0) ? (T)-val1 : val1;
        T abs_val2 = (val2 < (T)0) ? (T)-val2 : val2;
        working_storage[i] = (abs_val1 > abs_val2) ? abs_val1 : abs_val2;
    }

    // Stage 2: Process 4 elements -> 2 elements
    reduction_outer_loop: for (size_t i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        working_storage[i] = (working_storage[2 * i] > working_storage[2 * i + 1]) ? working_storage[2 * i] : working_storage[2 * i + 1];
    }

    // Stage 3: Process 2 elements -> 1 element (final result)
    return (working_storage[0] > working_storage[1]) ? working_storage[0] : working_storage[1];
}

#endif // __MAX_H__

