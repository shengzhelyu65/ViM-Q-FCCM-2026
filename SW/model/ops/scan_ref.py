import torch
import torch.nn.functional as F

from einops import rearrange, repeat
import numpy as np

def manual_scan(Lambda_elements, Bu_elements, padded_patches=200, vector_size=4, padded_vectors=64):
    # Shape:
    # Lambda_elements: [L, batch, dim, dstate]
    # Bu_elements: [L, batch, dim, dstate]
    
    num_vectors = padded_patches // vector_size
    
    l, b, d, n = Lambda_elements.shape
    data_type = Lambda_elements.dtype
    device = Lambda_elements.device
    
    indentity_gate = torch.ones((b, d, n), dtype=data_type, device=device)
    identity_token = torch.zeros((b, d, n), dtype=data_type, device=device)
    
    # Step 1: Pad the elements to padded_patches
    Lambda_elements_padded = torch.ones((padded_patches, b, d, n), dtype=data_type, device=device)
    Bu_elements_padded = torch.zeros((padded_patches, b, d, n), dtype=data_type, device=device)
    Lambda_elements_padded[:l] = Lambda_elements
    Bu_elements_padded[:l] = Bu_elements
    Lambda_elements_padded_group = Lambda_elements_padded.view((num_vectors, vector_size, b, d, n))
    Bu_elements_padded_group = Bu_elements_padded.view((num_vectors, vector_size, b, d, n))
    
    # Step 2: Perform the local scan
    local_scan_gates = torch.zeros((num_vectors, vector_size, b, d, n), dtype=data_type, device=device)
    local_scan_tokens = torch.zeros((num_vectors, vector_size, b, d, n), dtype=data_type, device=device)
    for i in range(num_vectors):
        for j in range(vector_size):
            if j == 0:
                (local_scan_gates[i, j], local_scan_tokens[i, j]) = binary_operator((indentity_gate, identity_token), (Lambda_elements_padded_group[i, j], Bu_elements_padded_group[i, j]))
            else:
                (local_scan_gates[i, j], local_scan_tokens[i, j]) = binary_operator((local_scan_gates[i, j - 1], local_scan_tokens[i, j - 1]), (Lambda_elements_padded_group[i, j], Bu_elements_padded_group[i, j]))
                
    # Step 3: Perform the global prefix scan
    partial_gates = torch.ones((padded_vectors, b, d, n), dtype=data_type, device=device)
    partial_tokens = torch.zeros((padded_vectors, b, d, n), dtype=data_type, device=device)
    for i in range(num_vectors):
        partial_gates[i] = local_scan_gates[i, -1]
        partial_tokens[i] = local_scan_tokens[i, -1]
    for i in range(num_vectors, padded_vectors):
        partial_gates[i] = indentity_gate
        partial_tokens[i] = identity_token

    (global_prefix_gates, global_prefix_tokens) = reduce_tree_scan((partial_gates, partial_tokens))

    # Step 4: Adjust local scan
    final_out_gates = torch.zeros((num_vectors, vector_size, b, d, n), dtype=data_type, device=device)
    final_out_tokens = torch.zeros((num_vectors, vector_size, b, d, n), dtype=data_type, device=device)
    for i in range(num_vectors):
        for j in range(vector_size):
            (final_out_gates[i, j], final_out_tokens[i, j]) = binary_operator((global_prefix_gates[i], global_prefix_tokens[i]), (local_scan_gates[i, j], local_scan_tokens[i, j]))
            
    # Step 5: Unpad the elements
    final_out_tokens_unpadded = final_out_tokens.view((padded_patches, b, d, n))[:l]
    
    return final_out_tokens_unpadded

def reduce_tree_scan(partial):
    (partial_gates, partial_tokens) = partial
    seq_len, _, _, _ = partial_gates.shape
    
    num_sweep = int(np.log2(seq_len))
    
    identity_gate = torch.ones_like(partial_gates[0])
    identity_token = torch.zeros_like(partial_tokens[0])
    
    # Up-sweep phase
    for d in range(0, num_sweep):
        for i in range(0, seq_len, 2 ** (d + 1)):
            left_idx = i + 2 ** d - 1
            right_idx = i + 2 ** (d + 1) - 1
            left_gate = partial_gates[left_idx]
            left_token = partial_tokens[left_idx]
            right_gate = partial_gates[right_idx]
            right_token = partial_tokens[right_idx]
            (partial_gates[right_idx], partial_tokens[right_idx]) = binary_operator((left_gate, left_token), (right_gate, right_token))
            
    # Clear the last element
    partial_gates[-1] = torch.ones_like(partial_gates[-1])
    partial_tokens[-1] = torch.zeros_like(partial_tokens[-1])
    
    # Down-sweep phase
    right_tmp_gate = torch.ones_like(partial_gates[-1])
    right_tmp_token = torch.zeros_like(partial_tokens[-1])
    for d in range(num_sweep - 1, -1, -1):
        for i in range(0, seq_len, 2 ** (d + 1)):
            left_idx = i + 2 ** d - 1
            right_idx = i + 2 ** (d + 1) - 1
            left_gate = partial_gates[left_idx]
            left_token = partial_tokens[left_idx]
            right_gate = partial_gates[right_idx]
            right_token = partial_tokens[right_idx]
            (right_tmp_gate, right_tmp_token) = binary_operator((identity_gate, identity_token), (right_gate, right_token))
            (partial_gates[right_idx], partial_tokens[right_idx]) = binary_operator((right_gate, right_token), (left_gate, left_token))
            partial_gates[left_idx] = right_tmp_gate
            partial_tokens[left_idx] = right_tmp_token
            
    return (partial_gates, partial_tokens)

def binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

def selective_scan_parallel(u, delta, A, B, C, D=None, z=None):
    """
    Selective scan with parallel up-sweep and down-sweep for efficient computation.
    """
    dtype_in = u.dtype
    u = u.float()

    delta = delta.float()
    delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]

    B = B.float()
    C = C.float()
    
    B = rearrange(B, "b 1 dstate l -> b dstate l")
    C = rearrange(C, "b 1 dstate l -> b dstate l")

    # Initialize latent space (state) to zeros
    x = A.new_zeros((batch, dim, dstate))  # Latent space initialized to zeros
    ys = []  # List to collect outputs for each timestep
    
    # Precompute deltaA = exp(delta * A) and deltaB_u = delta * B * u
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))  # [batch, dim, L, N]
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    lambda_elements = rearrange(deltaA, 'b d l n -> l b d n')
    bu_elements = rearrange(deltaB_u, 'b d l n -> l b d n')
    xs = manual_scan(lambda_elements, bu_elements) # [L, batch, dim, dstate]
    for i in range(u.shape[2]):
        x = xs[i]
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i]) # here: [batch, dim]
        ys.append(y)

    y = torch.stack(ys, dim=2) # [batch, dim, L]

    # Compute final output
    out = y + u * rearrange(D, "d -> d 1")
    out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out

## --------------------------- Reference implementations --------------------------- ##
def manual_scan_ref(Lambda_elements, Bu_elements):
    """Manually implement the parallel scan (prefix sum)"""
    # Initialize the accumulator with the first elements
    xs = [(Lambda_elements[0], Bu_elements[0])]
    
    # Perform the scan operation
    for i in range(1, Lambda_elements.shape[0]):
        new_state = binary_operator(xs[-1], (Lambda_elements[i], Bu_elements[i]))
        xs.append(new_state)
    
    return torch.stack([x[1] for x in xs], dim=0)

def reduce_tree_scan_ref(partial):
    seq_len, b, d, n = partial[0].shape
    data_type = partial[0].dtype
    device = partial[0].device
    identity_gate = torch.ones((b, d, n), dtype=data_type, device=device)
    identity_token = torch.zeros((b, d, n), dtype=data_type, device=device)
    (partial_gates, partial_tokens) = partial
    
    global_prefix_gates = torch.zeros((seq_len, b, d, n), dtype=data_type, device=device)
    global_prefix_tokens = torch.zeros((seq_len, b, d, n), dtype=data_type, device=device)
    global_prefix_gates[0] = identity_gate
    global_prefix_tokens[0] = identity_token
    for i in range(1, seq_len):
        (global_prefix_gates[i], global_prefix_tokens[i]) = binary_operator((global_prefix_gates[i - 1], global_prefix_tokens[i - 1]), (partial_gates[i - 1], partial_tokens[i - 1]))
    return (global_prefix_gates, global_prefix_tokens)