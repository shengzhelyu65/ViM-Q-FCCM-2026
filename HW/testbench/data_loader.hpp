#ifndef __DATA_LOADER_HPP__
#define __DATA_LOADER_HPP__

#include "../src/common.h"
#include "../src/tbutil.h"
#include <ap_int.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using std::ifstream;
using std::ostringstream;
using std::string;
using std::cerr;
using std::endl;

// Function to load reference data
template<typename T>
bool load_reference_data(const string& filename, T& data) {
    ifstream ifs(filename, std::ios::binary);
    read(ifs, data);
    if (!ifs) {
        cerr << "Error reading " << filename << endl;
        return false;
    }
    return true;
}

template<typename T>
bool load_data(const string& filename, T& data) {
    ifstream ifs(filename, std::ios::binary);
    read(ifs, data);
    if (!ifs)
    {
        cerr << "Error reading " << filename << endl;
        return false;
    }
    return true;
}

template<typename T>
bool load_layers_data(const string& filename, T& data, unsigned int num_layers) {
    FOR_EACH(layer, num_layers)
    {
        ostringstream oss;
        oss << "bin_float32_block/layers." << layer << "." << filename <<  ".bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, data[layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return false;
        }
    }
    return true;
}

inline bool load_int4_file_to_burst_words(
    const string& filename,
    wt_wide_t* dst_words,
    unsigned int total_elems
) {
    ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        cerr << "Error opening " << filename << endl;
        return false;
    }

    const unsigned int total_words = (total_elems + MAGS_PER_WORD - 1) / MAGS_PER_WORD;
    unsigned char cached_byte = 0;

    for (unsigned int w = 0; w < total_words; ++w) {
        wt_wide_t word = 0;
        for (unsigned int t = 0; t < MAGS_PER_WORD; ++t) {
            unsigned int idx = w * MAGS_PER_WORD + t;
            if (idx < total_elems) {
                // Read a new byte every 2 elements (even indices)
                if (idx % 2 == 0) {
                    char b;
                    ifs.read(&b, 1);
                    if (!ifs && !ifs.eof()) {
                        cerr << "Error reading at element index " << idx << " from " << filename << endl;
                        return false;
                    }
                    cached_byte = static_cast<unsigned char>(b);
                }
                
                ap_uint<4> v;
                if (idx % 2 == 0) {
                    // Even index: Upper nibble
                    v = (cached_byte >> 4) & 0x0F;
                } else {
                    // Odd index: Lower nibble
                    v = cached_byte & 0x0F;
                }
                word.range(4 * t + 3, 4 * t) = v;
            }
        }
        dst_words[w] = word;
    }
    return true;
}

inline bool load_bit_file_to_burst_words(
    const string& filename,
    wt_wide_t* dst_words,
    unsigned int total_bits
) {
    ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        cerr << "Error opening " << filename << endl;
        return false;
    }

    const unsigned int total_words = (total_bits + SIGNS_PER_WORD - 1) / SIGNS_PER_WORD;
    unsigned char cached_byte = 0;

    for (unsigned int w = 0; w < total_words; ++w) {
        wt_wide_t word = 0;
        for (unsigned int b = 0; b < SIGNS_PER_WORD; ++b) {
            unsigned int idx = w * SIGNS_PER_WORD + b;
            if (idx < total_bits) {
                // Read a new byte every 8 bits
                if (idx % 8 == 0) {
                    char val;
                    ifs.read(&val, 1);
                    if (!ifs && !ifs.eof()) {
                         cerr << "Error reading at bit index " << idx << " from " << filename << endl;
                         return false;
                    }
                    cached_byte = static_cast<unsigned char>(val);
                }
                
                unsigned int bit_pos = idx % 8;
                word[b] = (cached_byte >> bit_pos) & 0x01;
            }
        }
        dst_words[w] = word;
    }
    return true;
}

inline bool load_layers_weight_mags(
    wt_wide_t* dst_burst_mem,
    unsigned int total_elems,
    const string& base_filename = "mixer.in_proj.weight.magnitude"
) {
    unsigned int words_per_layer = total_elems / MAGS_PER_WORD;
    
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "bin_float32_block/layers." << layer << "." << base_filename << ".bin";
        string filename = oss.str();
        
        wt_wide_t* layer_dst = &dst_burst_mem[layer * words_per_layer];
        
        bool ok = load_int4_file_to_burst_words(filename, layer_dst, total_elems);
        if (!ok) {
            cerr << "Failed to load magnitudes from: " << filename << endl;
            return false;
        }
    }
    return true;
}

inline bool load_layers_weight_signs(
    wt_wide_t* dst_burst_mem,
    unsigned int total_bits,
    const string& base_filename = "mixer.in_proj.weight.sign"
) {
    unsigned int words_per_layer = total_bits / SIGNS_PER_WORD;
    
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "bin_float32_block/layers." << layer << "." << base_filename << ".bin";
        string filename = oss.str();
        
        wt_wide_t* layer_dst = &dst_burst_mem[layer * words_per_layer];
        
        bool ok = load_bit_file_to_burst_words(filename, layer_dst, total_bits);
        if (!ok) {
            cerr << "Failed to load signs from: " << filename << endl;
            return false;
        }
    }
          return true;
  }
  
  #endif // __DATA_LOADER_HPP__ 