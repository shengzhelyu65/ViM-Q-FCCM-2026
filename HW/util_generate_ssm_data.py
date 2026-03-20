import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from constants import *

def generate_ssm_data(layer_idx=0, seed=42):
    np.random.seed(seed)
    
    # u and z_silu are usually activations, range around [-1.5, 1.5]
    u = np.random.randn(NUM_PATCHES, D_INNER).astype(np.float32) * 0.8
    z_silu = np.random.randn(NUM_PATCHES, D_INNER).astype(np.float32) * 0.8
    
    # delta is usually positive, range [0.001, 0.1]
    # We use uniform to keep it in a safe range
    delta = np.random.uniform(0.001, 0.04, size=(NUM_PATCHES, D_INNER)).astype(np.float32)
    
    # B and C are projection outputs, range around [-0.5, 0.5]
    B = np.random.randn(NUM_PATCHES, D_STATE).astype(np.float32) * 0.2
    C = np.random.randn(NUM_PATCHES, D_STATE).astype(np.float32) * 0.2
    
    # A is typically negative in Mamba, range [-8, -0.1]
    # We use a more realistic range that fits the fast_exp_approx stability
    A = -np.random.uniform(0.1, 3.0, size=(SCAN_DIM,)).astype(np.float32)
    
    # D is a skip connection weight, range [0, 0.5]
    D = np.random.rand(D_INNER).astype(np.float32) * 0.5
    
    return u, delta, z_silu, B, C, A, D

def compute_ssm_cpu_reference(u, delta, z_silu, B, C, A, D):
    from math import ceil
    
    FEATURE_BLOCK_SIZE = 8
    NUM_FEATURE_BLOCKS_SCAN = ceil(SCAN_DIM / FEATURE_BLOCK_SIZE)
    state_dim_iters = ceil(D_STATE / FEATURE_BLOCK_SIZE)
    inner_dim_blocks = ceil(D_INNER / FEATURE_BLOCK_SIZE)
    total_dim_iters = D_INNER * state_dim_iters
    read_iters = FEATURE_BLOCK_SIZE * state_dim_iters

    A_2d = A.reshape(D_INNER, D_STATE)
    
    output = np.zeros((NUM_PATCHES, D_INNER), dtype=np.float32)
    
    for p in range(NUM_PATCHES):
        pe_state_token = np.zeros((NUM_FEATURE_BLOCKS_SCAN, FEATURE_BLOCK_SIZE), dtype=np.float32)
        
        next_total_dim_block = 0
        next_state_dim_block = 0
        next_inner_dim_block = 0
        
        delta_u_block_idx = 0
        B_block_idx = 0
        
        for scan_iter in range(NUM_FEATURE_BLOCKS_SCAN):
            total_dim_block = next_total_dim_block
            state_dim_block = next_state_dim_block
            inner_dim_block = next_inner_dim_block
            
            next_total_dim_block = (total_dim_block + 1) % total_dim_iters
            next_state_dim_block = (state_dim_block + 1) % state_dim_iters
            if total_dim_block == total_dim_iters - 1:
                next_inner_dim_block = 0
            elif state_dim_block == state_dim_iters - 1:
                next_inner_dim_block = inner_dim_block + 1
            else:
                next_inner_dim_block = inner_dim_block
            
            inner_dim_offset = inner_dim_block % FEATURE_BLOCK_SIZE
            inner_dim = inner_dim_block
            
            if total_dim_block % read_iters == 0:
                delta_block_start = delta_u_block_idx * FEATURE_BLOCK_SIZE
                delta_block = delta[p, delta_block_start:min(delta_block_start + FEATURE_BLOCK_SIZE, D_INNER)]
                u_block = u[p, delta_block_start:min(delta_block_start + FEATURE_BLOCK_SIZE, D_INNER)]
                delta_u_block_idx = (delta_u_block_idx + 1) % inner_dim_blocks
                if len(delta_block) < FEATURE_BLOCK_SIZE:
                    delta_block = np.pad(delta_block, (0, FEATURE_BLOCK_SIZE - len(delta_block)), 'constant')
                if len(u_block) < FEATURE_BLOCK_SIZE:
                    u_block = np.pad(u_block, (0, FEATURE_BLOCK_SIZE - len(u_block)), 'constant')
            
            if inner_dim_block == 0:
                B_block_start = B_block_idx * FEATURE_BLOCK_SIZE
                B_block = B[p, B_block_start:min(B_block_start + FEATURE_BLOCK_SIZE, D_STATE)]
                B_block_idx = (B_block_idx + 1) % state_dim_iters
                if len(B_block) < FEATURE_BLOCK_SIZE:
                    B_block = np.pad(B_block, (0, FEATURE_BLOCK_SIZE - len(B_block)), 'constant')
            
            inner_dim_for_A = total_dim_block // state_dim_iters
            state_dim_for_A = total_dim_block % state_dim_iters
            A_block_start = inner_dim_for_A * D_STATE + state_dim_for_A * FEATURE_BLOCK_SIZE
            A_slice = A[A_block_start:min(A_block_start + FEATURE_BLOCK_SIZE, SCAN_DIM)]
            if len(A_slice) < FEATURE_BLOCK_SIZE:
                A_slice = np.pad(A_slice, (0, FEATURE_BLOCK_SIZE - len(A_slice)), 'constant')
            
            delta_val = delta_block[inner_dim_offset]
            u_val = u_block[inner_dim_offset]
            
            delta_A_block = np.exp(delta_val * A_slice)
            delta_Bu_block = delta_val * B_block * u_val
            
            scan_block_idx = scan_iter % NUM_FEATURE_BLOCKS_SCAN
            current_token = pe_state_token[scan_block_idx, :]
            new_token = current_token * delta_A_block + delta_Bu_block
            pe_state_token[scan_block_idx, :] = new_token
        
        x_C = np.zeros(D_INNER, dtype=np.float32)
        
        next_total_dim_block = 0
        next_state_dim_block = 0
        next_inner_dim_block = 0
        next_out_dim_block = 0
        next_C_src_block = 0
        tmp_res_reg = 0.0
        
        for scan_iter in range(NUM_FEATURE_BLOCKS_SCAN):
            total_dim_block = next_total_dim_block
            state_dim_block = next_state_dim_block
            inner_dim_block = next_inner_dim_block
            out_dim_block = next_out_dim_block
            C_src_block = next_C_src_block
            
            next_total_dim_block = (total_dim_block + 1) % total_dim_iters
            next_state_dim_block = (state_dim_block + 1) % state_dim_iters
            if total_dim_block == total_dim_iters - 1:
                next_inner_dim_block = 0
            elif state_dim_block == state_dim_iters - 1:
                next_inner_dim_block = inner_dim_block + 1
            else:
                next_inner_dim_block = inner_dim_block
            
            if state_dim_block == state_dim_iters - 1:
                next_out_dim_block = (out_dim_block + 1) % FEATURE_BLOCK_SIZE
            else:
                next_out_dim_block = out_dim_block
            
            if inner_dim_block == 0:
                next_C_src_block = (C_src_block + 1) % state_dim_iters
            else:
                next_C_src_block = C_src_block
            
            scan_block_idx = scan_iter % NUM_FEATURE_BLOCKS_SCAN
            scan_output = pe_state_token[scan_block_idx, :]
            
            if inner_dim_block == 0:
                C_block_start = C_src_block * FEATURE_BLOCK_SIZE
                C_block = C[p, C_block_start:min(C_block_start + FEATURE_BLOCK_SIZE, D_STATE)]
                if len(C_block) < FEATURE_BLOCK_SIZE:
                    C_block = np.pad(C_block, (0, FEATURE_BLOCK_SIZE - len(C_block)), 'constant')
            
            if state_dim_block == 0:
                tmp_res = 0.0
            else:
                tmp_res = tmp_res_reg
            
            partial_products = scan_output * C_block
            current_sum = np.sum(partial_products)
            tmp_res += current_sum
            tmp_res_reg = tmp_res
            
            if out_dim_block == FEATURE_BLOCK_SIZE - 1 and state_dim_block == state_dim_iters - 1:
                inner_dim = inner_dim_block
                if inner_dim < D_INNER:
                    x_C[inner_dim] = tmp_res
        
        for i in range(D_INNER):
            output[p, i] = (x_C[i] + u[p, i] * D[i]) * z_silu[p, i]
    
    return output

def save_binary(data, filename):
    """Save numpy array to binary file."""
    with open(filename, 'wb') as f:
        data.tofile(f)

def main():
    layer_idx = 0
    if len(sys.argv) > 1:
        layer_idx = int(sys.argv[1])
    
    print(f"Generating SSM test data for layer {layer_idx}...")
    
    u, delta, z_silu, B, C, A, D = generate_ssm_data(layer_idx)
    
    print("Computing CPU reference...")
    output_ref = compute_ssm_cpu_reference(u, delta, z_silu, B, C, A, D)
    
    project_root = os.path.dirname(__file__)
    scan_data_dir = os.path.join(project_root, 'data', 'ssm_float32')
    os.makedirs(scan_data_dir, exist_ok=True)
    
    save_binary(u, os.path.join(scan_data_dir, f'layers.{layer_idx}.mixer.scan.u.float32.bin'))
    save_binary(delta, os.path.join(scan_data_dir, f'layers.{layer_idx}.mixer.scan.delta.float32.bin'))
    save_binary(z_silu, os.path.join(scan_data_dir, f'layers.{layer_idx}.mixer.scan.z_silu.float32.bin'))
    save_binary(B, os.path.join(scan_data_dir, f'layers.{layer_idx}.mixer.scan.B.float32.bin'))
    save_binary(C, os.path.join(scan_data_dir, f'layers.{layer_idx}.mixer.scan.C.float32.bin'))
    save_binary(A, os.path.join(scan_data_dir, f'layers.{layer_idx}.mixer.scan.A.float32.bin'))
    save_binary(D, os.path.join(scan_data_dir, f'layers.{layer_idx}.mixer.scan.D.float32.bin'))
    save_binary(output_ref, os.path.join(scan_data_dir, f'layers.{layer_idx}.mixer.scan.output.float32.bin'))
    
    print(f"Generated test data:")
    print(f"  u: {u.shape}")
    print(f"  delta: {delta.shape}")
    print(f"  z_silu: {z_silu.shape}")
    print(f"  B: {B.shape}")
    print(f"  C: {C.shape}")
    print(f"  A: {A.shape}")
    print(f"  D: {D.shape}")
    print(f"  output: {output_ref.shape}")
    print(f"  Output range: [{output_ref.min():.6f}, {output_ref.max():.6f}]")
    print(f"  Output mean: {output_ref.mean():.6f}, std: {output_ref.std():.6f}")
    print(f"Data saved to: {scan_data_dir}")
    print("Done!")

if __name__ == '__main__':
    main()
