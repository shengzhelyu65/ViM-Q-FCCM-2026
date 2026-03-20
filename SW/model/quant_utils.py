import torch
import math

@torch.no_grad()
def round_to_power_of_2(x):
    """Round values to the nearest power of 2."""
    log2 = torch.log2(x)
    rounded_log2 = torch.round(log2)
    return torch.pow(2, rounded_log2)

@torch.no_grad()
def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))

    values = []
    int_weights = []
    
    # Allocate bits efficiently
    bits_a = max(1, math.ceil(math.log2(len(base_a))))
    bits_b = max(1, math.ceil(math.log2(len(base_b))))
    bits_c = max(1, math.ceil(math.log2(len(base_c)))) if len(base_c) > 1 else 0
    
    for i, a in enumerate(base_a):
        for j, b in enumerate(base_b):
            for k, c in enumerate(base_c):
                values.append((a + b + c))
                # Pack into B bits: base_a_idx | (base_b_idx << bits_a) | (base_c_idx << (bits_a + bits_b))
                int_weight = i | (j << bits_a) | (k << (bits_a + bits_b))
                int_weight = int_weight & ((1 << B) - 1)  # Mask to B bits
                int_weights.append(int_weight)
    
    # Remove duplicates while preserving correspondence and ensuring determinism
    unique_pairs = sorted(list(set(zip(values, int_weights))))
    values, int_weights = zip(*unique_pairs)
    
    values = torch.Tensor(list(values))
    value_scale = 1.0 / torch.max(values)
    values = values.mul(value_scale)
    int_weights = torch.tensor(list(int_weights), dtype=torch.uint8)
    
    # Return bases as tensors along with values and int_weights
    base_a_tensor = torch.tensor(base_a, dtype=torch.float32)
    base_b_tensor = torch.tensor(base_b, dtype=torch.float32)
    base_c_tensor = torch.tensor(base_c, dtype=torch.float32)
    
    return values, value_scale, int_weights, base_a_tensor, base_b_tensor, base_c_tensor, (bits_a, bits_b, bits_c)
