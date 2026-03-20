#ifndef __TB_UTILS_HPP__
#define __TB_UTILS_HPP__

#include "../src/common.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

using std::ifstream;
using std::ostream;
using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::flush;
using std::fixed;
using std::left;
using std::setprecision;
using std::setw;

// Constants for display limits
constexpr unsigned int DISPLAY_PATCH_LIMIT = 10;
constexpr unsigned int DISPLAY_DIM_LIMIT = 6;

#include <cstdlib>

// Get project root directory - read from environment variable VIM_Q_HW_ROOT
// Usage: export VIM_Q_HW_ROOT=/path/to/ViM_Q/HW
inline const char* get_project_root() {
    const char* env_hw_root = std::getenv("VIM_Q_HW_ROOT");
    if (env_hw_root == nullptr) {
        cerr << "ERROR: VIM_Q_HW_ROOT environment variable is not set. Please export VIM_Q_HW_ROOT=/path/to/ViM_Q/HW" << endl;
        std::exit(1);
    }
    return env_hw_root;
}

typedef hls::vector<float, FEATURE_BLOCK_SIZE> fm_block_t_fp;
typedef fm_block_t_fp fm_blocks_t_fp[ceildiv(D_MODEL, FEATURE_BLOCK_SIZE)];
typedef fm_blocks_t_fp patch_blocks_t_fp[NUM_PATCHES];
typedef fm_block_t_fp fm_blocks_inner_t_fp[ceildiv(D_INNER, FEATURE_BLOCK_SIZE)];
typedef fm_blocks_inner_t_fp patch_blocks_inner_t_fp[NUM_PATCHES];
typedef fm_block_t_fp fm_blocks_state_t_fp[ceildiv(D_STATE, FEATURE_BLOCK_SIZE)];
typedef fm_blocks_state_t_fp patch_blocks_state_t_fp[NUM_PATCHES];
typedef fm_block_t_fp fm_blocks_scan_t_fp[ceildiv(SCAN_DIM, FEATURE_BLOCK_SIZE)];
typedef fm_blocks_scan_t_fp patch_blocks_scan_t_fp[NUM_PATCHES];
typedef fm_block_t_fp final_result_t_fp[ceildiv(PADDED_NUM_CLASSES, FEATURE_BLOCK_SIZE)];
typedef fm_block_t_fp patch_cls_token_t_fp[1];

// Structure to hold comparison results
struct ComparisonResult {
    float mse;
    float mae;
    float mse_cls;
    float mae_cls;
    string description;
};

// Function to display comparison results
void display_comparison_result(const ComparisonResult& result) {
    cout << result.description << ":" << endl;
    cout << "  MSE: " << result.mse << " (all patches)" << endl;
    cout << "  MAE: " << result.mae << " (all patches)" << endl;
    cout << "  MSE (CLS): " << result.mse_cls << " (patch " << CLS_TOKEN_IDX << ")" << endl;
    cout << "  MAE (CLS): " << result.mae_cls << " (patch " << CLS_TOKEN_IDX << ")" << endl;
    cout << endl;
}

// Template function to display sample values - specialization for cls token (single entry)
template<typename T, typename RefT>
void display_sample_values_cls(const T& computed, const RefT& reference, const string& title) {
    cout << "Sample of values from " << title << ":" << endl;
    ostream formatted(cout.rdbuf());
    formatted << setprecision(8) << fixed;
    
    // Display computed values for cls token (index 0)
    cout << "[[";
    unsigned int dim_blocks = (DISPLAY_DIM_LIMIT + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE;
    for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
        for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
            unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
            if (dim >= DISPLAY_DIM_LIMIT) break;
            
            float comp_val = computed[0][dim_block][dim_offset].to_float();
            if (comp_val >= 0.0) cout << " ";
            formatted << setw(9 + (comp_val < 0.0)) << left << comp_val;
            if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
        }
    }
    cout << "]]    ";
    
    // Display reference values for cls token (index 0)
    cout << "[[";
    for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
        for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
            unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
            if (dim >= DISPLAY_DIM_LIMIT) break;
            
            float ref_val = reference[0][dim_offset];  // reference is fm_block_t_fp[1] 
            if (ref_val >= 0.0) cout << " ";
            formatted << setw(9 + (ref_val < 0.0)) << left << ref_val;
            if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
        }
    }
    cout << "]]" << endl << endl;
}

// Template function to display sample values
template<typename T, typename RefT>
void display_sample_values(const T& computed, const RefT& reference, const string& title, 
                          unsigned int num_patches, 
                          unsigned int cls_token_idx) {
    cout << "Sample of values from " << title << ":" << endl;
    ostream formatted(cout.rdbuf());
    formatted << setprecision(8) << fixed;
    
    unsigned int max_patches = (num_patches < DISPLAY_PATCH_LIMIT) ? num_patches : DISPLAY_PATCH_LIMIT;
    
    for (unsigned int patch = 0; patch < max_patches; patch++) {
        // Display computed values
        cout << ((patch == 0) ? "[[" : " [");
        unsigned int dim_blocks = (DISPLAY_DIM_LIMIT + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE;
        for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
            for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
                unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
                if (dim >= DISPLAY_DIM_LIMIT) break;
                
                float comp_val = computed[patch][dim_block][dim_offset].to_float();
                if (comp_val >= 0.0) cout << " ";
                formatted << setw(9 + (comp_val < 0.0)) << left << comp_val;
                if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
            }
        }
        cout << ((patch == max_patches - 1) ? "]]" : "],") << "    ";
        
        // Display reference values
        cout << ((patch == 0) ? "[[" : " [");
        for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
            for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
                unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
                if (dim >= DISPLAY_DIM_LIMIT) break;
                
                float ref_val = reference[patch][dim_block][dim_offset];
                if (ref_val >= 0.0) cout << " ";
                formatted << setw(9 + (ref_val < 0.0)) << left << ref_val;
                if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
            }
        }
        cout << ((patch == max_patches - 1) ? "]]" : "],") << endl;
    }
    cout << endl;
}

// Template function to compare results - specialization for cls token
template<typename T, typename RefT>
ComparisonResult compare_results_cls(const T& computed, const RefT& reference, const string& description,
                                   unsigned int dimension) {
    ComparisonResult result;
    result.description = description;
    result.mse = 0.0;
    result.mae = 0.0;
    result.mse_cls = 0.0;
    result.mae_cls = 0.0;
    
    unsigned int dim_blocks = (dimension + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE;
    for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
        for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
            unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
            if (dim >= dimension) break;
            
            float computed_val = computed[0][dim_block][dim_offset].to_float();
            float actual_val = reference[0][dim_offset];  // reference is fm_block_t_fp[1]
            float error = actual_val - computed_val;
            float abs_error = (error < 0.0) ? -error : error;
            
            result.mse += error * error;
            result.mae += abs_error;
            result.mse_cls += error * error;
            result.mae_cls += abs_error;
        }
    }
    
    result.mse /= static_cast<float>(dimension);
    result.mae /= static_cast<float>(dimension);
    result.mse_cls /= static_cast<float>(dimension);
    result.mae_cls /= static_cast<float>(dimension);
    
    return result;
}

// Template function to compare results
template<typename T, typename RefT>
ComparisonResult compare_results(const T& computed, const RefT& reference, const string& description,
                               unsigned int num_patches, unsigned int dimension, 
                               unsigned int cls_token_idx) {
    ComparisonResult result;
    result.description = description;
    result.mse = 0.0;
    result.mae = 0.0;
    result.mse_cls = 0.0;
    result.mae_cls = 0.0;
    
    // Track statistics for verification
    unsigned int cls_count = 0;
    unsigned int total_count = 0;
    
    for (unsigned int patch = 0; patch < num_patches; patch++) {
        unsigned int dim_blocks = (dimension + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE;
        for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
            for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
                unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
                if (dim >= dimension) break;
                
                float computed_val = computed[patch][dim_block][dim_offset].to_float();
                float actual_val = reference[patch][dim_block][dim_offset];
                float error = actual_val - computed_val;
                float abs_error = (error < 0.0) ? -error : error;
                
                result.mse += error * error;
                result.mae += abs_error;
                total_count++;
                
                if (patch == cls_token_idx) {
                    result.mse_cls += error * error;
                    result.mae_cls += abs_error;
                    cls_count++;
                }
            }
        }
    }
    
    // Calculate averages
    float total_elements = static_cast<float>(total_count);
    float cls_elements = static_cast<float>(cls_count);
    
    result.mse /= total_elements;
    result.mae /= total_elements;
    result.mse_cls /= cls_elements;
    result.mae_cls /= cls_elements;
    
    // Verify: total_count should be num_patches * dimension
    // cls_count should be dimension
    // This helps catch any indexing bugs
    
    return result;
}

// Function to perform complete comparison for cls token (display + compare + print results)
template<typename T, typename RefT>
ComparisonResult perform_complete_comparison_cls(const T& computed, const RefT& reference,
                                               const string& title, const string& description,
                                               unsigned int dimension) {
    display_sample_values_cls(computed, reference, title);
    ComparisonResult result = compare_results_cls(computed, reference, description, dimension);
    display_comparison_result(result);
    return result;
}

// Template function to display sample values for final result (single classification result)
template<typename T, typename RefT>
void display_sample_values_final(const T& computed, const RefT& reference, const string& title) {
    cout << "Sample of values from " << title << ":" << endl;
    ostream formatted(cout.rdbuf());
    formatted << setprecision(8) << fixed;
    
    // Display computed values for final result
    cout << "[[";
    unsigned int dim_blocks = (DISPLAY_DIM_LIMIT + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE;
    for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
        for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
            unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
            if (dim >= DISPLAY_DIM_LIMIT) break;
            
            float comp_val = computed[dim_block][dim_offset].to_float();
            if (comp_val >= 0.0) cout << " ";
            formatted << setw(9 + (comp_val < 0.0)) << left << comp_val;
            if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
        }
    }
    cout << "]]    ";
    
    // Display reference values for final result
    cout << "[[";
    for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
        for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
            unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
            if (dim >= DISPLAY_DIM_LIMIT) break;
            
            float ref_val = reference[dim_block][dim_offset];
            if (ref_val >= 0.0) cout << " ";
            formatted << setw(9 + (ref_val < 0.0)) << left << ref_val;
            if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
        }
    }
    cout << "]]" << endl << endl;
}

// Template function to compare results for final result
template<typename T, typename RefT>
ComparisonResult compare_results_final(const T& computed, const RefT& reference, const string& description,
                                     unsigned int dimension) {
    ComparisonResult result;
    result.description = description;
    result.mse = 0.0;
    result.mae = 0.0;
    result.mse_cls = 0.0;
    result.mae_cls = 0.0;
    
    unsigned int dim_blocks = (dimension + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE;
    for (unsigned int dim_block = 0; dim_block < dim_blocks; dim_block++) {
        for (unsigned int dim_offset = 0; dim_offset < FEATURE_BLOCK_SIZE; dim_offset++) {
            unsigned int dim = dim_block * FEATURE_BLOCK_SIZE + dim_offset;
            if (dim >= dimension) break;
            
            float computed_val = computed[dim_block][dim_offset].to_float();
            float actual_val = reference[dim_block][dim_offset];
            float error = actual_val - computed_val;
            float abs_error = (error < 0.0) ? -error : error;
            
            result.mse += error * error;
            result.mae += abs_error;
            result.mse_cls += error * error;
            result.mae_cls += abs_error;
        }
    }
    
    result.mse /= static_cast<float>(dimension);
    result.mae /= static_cast<float>(dimension);
    result.mse_cls /= static_cast<float>(dimension);
    result.mae_cls /= static_cast<float>(dimension);
    
    return result;
}

// Function to perform complete comparison for final result
template<typename T, typename RefT>
ComparisonResult perform_complete_comparison_final(const T& computed, const RefT& reference,
                                                 const string& title, const string& description,
                                                 unsigned int dimension) {
    display_sample_values_final(computed, reference, title);
    ComparisonResult result = compare_results_final(computed, reference, description, dimension);
    display_comparison_result(result);
    return result;
}

// Function to perform complete comparison (display + compare + print results)
template<typename T, typename RefT>
ComparisonResult perform_complete_comparison(const T& computed, const RefT& reference,
                                           const string& title, const string& description,
                                           unsigned int num_patches, unsigned int dimension,
                                           unsigned int cls_token_idx) {
    display_sample_values(computed, reference, title, num_patches, cls_token_idx);
    ComparisonResult result = compare_results(computed, reference, description, num_patches, dimension, cls_token_idx);
    display_comparison_result(result);
    return result;
}

#endif // __TB_UTILS_HPP__ 