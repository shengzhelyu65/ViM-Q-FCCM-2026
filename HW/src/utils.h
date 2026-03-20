#ifndef __INT_UTILS_H__
#define __INT_UTILS_H__

#include "common.h"
#include <vector>
#include <fstream>
#include <string>
#include <cassert>

using namespace std;

/**
 * @brief Load binary data from file into a vector
 * @param filename Path to the binary file
 * @param data Vector to store the loaded data
 * @return true if successful, false otherwise
 */
template<class data_t>
bool load_binary_file(const string& filename, vector<data_t>& data) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return false;
    }
    
    // Get file size
    file.seekg(0, ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);
    
    // Read data
    size_t num_elements = file_size / sizeof(data_t);
    data.resize(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();
    
    return true;
}

/**
 * @brief Convert tensor (vector) to array with specified dimensions
 * @param tensor Input tensor vector
 * @param array Output array
 * @param H_LOAD Loaded height dimension
 * @param H Actual height dimension
 * @param T_LOAD Loaded time dimension
 * @param T Actual time dimension
 * @param C_LOAD Loaded channel dimension
 * @param C Actual channel dimension
 */
template<class data_t>
void tensor2array(
    vector<data_t>& tensor,
    data_t array[],
    int H_LOAD,
    int H,
    int T_LOAD,
    int T,
    int C_LOAD,
    int C
) {
    assert(tensor.size() == H_LOAD * T_LOAD * C_LOAD);
    for(int h=0; h<H; ++h){
        for(int t=0; t<T; ++t){
            for(int c=0; c<C; ++c){
                array[h*T*C + t*C + c] = tensor[h*T_LOAD*C_LOAD + t*C_LOAD + c];
            }
        }
    }
}

/**
 * @brief Convert array to stream for HLS processing
 * @param stream Output stream
 * @param array Input array
 * @param H Height dimension
 * @param T Time dimension
 * @param C Channel dimension
 */
template<class data_t, int BLOCK_SIZE>
void array2stream(
    hls::stream<hls::vector<data_t, BLOCK_SIZE>>& stream,
    data_t array[],
    int H,
    int T,
    int C
) {
    constexpr int CT = C / BLOCK_SIZE;
    for(int h=0; h<H; ++h){
        for(int t=0; t<T; ++t){
            for(int ct=0; ct<CT; ++ct){
                hls::vector<data_t, BLOCK_SIZE> block;
                for(int c=0; c<BLOCK_SIZE; ++c){
                    block[c] = array[h*T*C + t*C + ct*BLOCK_SIZE + c];
                }
                stream << block;
            }
        }
    }
}

/**
 * @brief Compare two arrays with tolerance
 * @param array1 First array
 * @param array2 Second array
 * @param size Size of arrays
 * @param tolerance Tolerance for comparison
 * @return true if arrays match within tolerance
 */
template<class data_t>
bool compare_arrays(
    data_t array1[],
    data_t array2[],
    int size,
    double tolerance = 1e-3
) {
    bool match = true;
    for(int i=0; i<size; ++i){
        double diff = abs((double)array1[i] - (double)array2[i]);
        if(diff > tolerance){
            cerr << "Mismatch at index " << i << ": " << array1[i] << " vs " << array2[i] << endl;
            match = false;
        }
    }
    return match;
}

#endif

