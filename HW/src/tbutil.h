#ifndef __LOAD_UTIL_H__
#define __LOAD_UTIL_H__

#include <istream>
#include <ostream>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <cstdint>

inline std::istream& read(std::istream& stream, bool& value)
{
    char byte;
    stream.read(&byte, 1);
    value = (byte != 0);
    return stream;
}

inline std::istream& read(std::istream& stream, float& value)
{
    stream.read(reinterpret_cast<char*>(&value), sizeof(float));
    return stream;
}

template<int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
inline std::istream& read(std::istream& stream, ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& value)
{
    union {
        float value;
        char bytes[sizeof(float)];
    } readable;
    stream.read(readable.bytes, sizeof(float));
    value = readable.value;
    return stream;
}

template<int _AP_W>
inline std::istream& read(std::istream& stream, ap_uint<_AP_W>& value)
{
    union {
        int8_t int_value;  // Changed to int8_t to read signed 8-bit integers
        char bytes[sizeof(int8_t)];
    } readable;
    stream.read(readable.bytes, sizeof(int8_t));

    // Convert int8 to ap_uint<_AP_W> by taking the lower _AP_W bits
    value = static_cast<ap_uint<_AP_W>>(readable.int_value & ((1 << _AP_W) - 1));
    
    return stream;
}

template<typename T, size_t N>
inline std::istream& read(std::istream& stream, hls::vector<T, N>& vector)
{
    for (size_t i = 0; i < N; i++)
    {
        read(stream, vector[i]);
    }
    return stream;
}

template<typename T, size_t N>
inline std::istream& read(std::istream& stream, T(&array)[N])
{
    for (size_t i = 0; i < N; i++)
    {
        read(stream, array[i]);
    }
    return stream;
}

#endif
