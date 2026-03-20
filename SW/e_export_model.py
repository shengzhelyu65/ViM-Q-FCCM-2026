import functools
import math
import os
import argparse
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from timm.models import create_model
from datasets import build_dataset
from engine import evaluate
import utils as utils
from einops import rearrange

from b_smooth_model import smooth_vim_model
from c_quantize_model import quantize_vim_model

import models_mamba
from model.mamba_simple_module import MambaModule
from model.quant_linear import QuantLinear
from model.quant_conv1d import QuantConv1d
from models_mamba import Block

def reorder_weights_to_blocks(weights, out_dim, in_dim, block_size=16):
    ## Determine if padding is needed
    if out_dim % block_size != 0 or in_dim % block_size != 0:
        assert False, f"Padding is needed for {out_dim}x{in_dim} and block size {block_size}"
        return weights
    
    # Calculate number of blocks
    num_out_blocks = out_dim // block_size
    num_in_blocks = in_dim // block_size
    
    # Reshape: (out_dim, in_dim) -> (out_dim//block, block, in_dim//block, block)
    reshaped = weights.reshape(num_out_blocks, block_size, num_in_blocks, block_size)
    
    # Permute: (0, 2, 1, 3) -> (out_dim//block, in_dim//block, block, block)
    permuted = reshaped.transpose(0, 2, 1, 3)
    
    # Combine: (0x2, 1, 3) -> (out_dim//block * in_dim//block, block, block)
    reordered_weights = permuted.reshape(num_out_blocks * num_in_blocks, block_size, block_size)
    
    return reordered_weights

def save_intermediate_outputs(model, dataloader, save_dir_base, device=None, export_dtype=torch.float32):
    """Save intermediate inputs and outputs for norm, QuantLinear and QuantConv1d layers"""
    dtype_text = "float32" if export_dtype == torch.float32 else "float16"
    np_dtype = np.float32 if export_dtype == torch.float32 else np.float16
    save_folder = os.path.join(save_dir_base, f"ref_{dtype_text}_block")
    image_folder = os.path.join(save_dir_base, f"image_{dtype_text}_block")
    
    # Clean and create directories
    for folder in [save_folder, image_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    
    model.eval()
    
    hooks = []
    final_norm_capture = {'hidden': None, 'residual': None}
    
    def hook_fn(module, input, output, name):
        """Hook function to capture both inputs and outputs of target layers"""
        # Save input tensor
        if len(input) >= 1:
            input_tensor = input[0]
            if input_tensor is not None:
                tensor_np = input_tensor.to(export_dtype).detach().cpu().numpy()
                if isinstance(module, QuantConv1d) and tensor_np.ndim == 3:
                    tensor_np = tensor_np.transpose(0, 2, 1)
                tensor_np.tofile(f"{save_folder}/{name}.input.{dtype_text}.bin")
                
        if isinstance(module, MambaModule):
            if hasattr(module, 'last_delta') and module.last_delta is not None:
                delta_np = module.last_delta.to(export_dtype).detach().cpu().numpy()
                delta_np.tofile(f"{save_folder}/{name}.delta.{dtype_text}.bin")
            
            if hasattr(module, 'last_B') and module.last_B is not None:
                B_np = module.last_B.to(export_dtype).detach().cpu().numpy()
                B_np.tofile(f"{save_folder}/{name}.B.{dtype_text}.bin")
            
            if hasattr(module, 'last_C') and module.last_C is not None:
                C_np = module.last_C.to(export_dtype).detach().cpu().numpy()
                C_np.tofile(f"{save_folder}/{name}.C.{dtype_text}.bin")
            
            if hasattr(module, 'last_lambda_elements') and module.last_lambda_elements is not None:
                lambda_np = module.last_lambda_elements.to(export_dtype).detach().cpu().numpy()
                lambda_np.tofile(f"{save_folder}/{name}.deltaA.{dtype_text}.bin")
                
            if hasattr(module, 'last_bu_elements') and module.last_bu_elements is not None:
                bu_np = module.last_bu_elements.to(export_dtype).detach().cpu().numpy()
                bu_np.tofile(f"{save_folder}/{name}.deltaBu.{dtype_text}.bin")
                
            if hasattr(module, 'last_xs') and module.last_xs is not None:
                xs_np = module.last_xs.to(export_dtype).detach().cpu().numpy()
                xs_np.tofile(f"{save_folder}/{name}.xs.{dtype_text}.bin")

            if hasattr(module, 'last_out_z') and module.last_out_z is not None:
                out_z_np = module.last_out_z.to(export_dtype).detach().cpu().numpy()
                out_z_np.tofile(f"{save_folder}/{name}.out_z.{dtype_text}.bin")
            
            # Save backward pass intermediate values
            if hasattr(module, 'last_delta_b') and module.last_delta_b is not None:
                delta_b_np = module.last_delta_b.to(export_dtype).detach().cpu().numpy()
                delta_b_np.tofile(f"{save_folder}/{name}.delta_b.{dtype_text}.bin")
            
            if hasattr(module, 'last_B_b') and module.last_B_b is not None:
                B_b_np = module.last_B_b.to(export_dtype).detach().cpu().numpy()
                B_b_np.tofile(f"{save_folder}/{name}.B_b.{dtype_text}.bin")
            
            if hasattr(module, 'last_C_b') and module.last_C_b is not None:
                C_b_np = module.last_C_b.to(export_dtype).detach().cpu().numpy()
                C_b_np.tofile(f"{save_folder}/{name}.C_b.{dtype_text}.bin")
            
            if hasattr(module, 'last_lambda_elements_b') and module.last_lambda_elements_b is not None:
                lambda_b_np = module.last_lambda_elements_b.to(export_dtype).detach().cpu().numpy()
                lambda_b_np.tofile(f"{save_folder}/{name}.deltaA_b.{dtype_text}.bin")
                
            if hasattr(module, 'last_bu_elements_b') and module.last_bu_elements_b is not None:
                bu_b_np = module.last_bu_elements_b.to(export_dtype).detach().cpu().numpy()
                bu_b_np.tofile(f"{save_folder}/{name}.deltaBu_b.{dtype_text}.bin")
                
            if hasattr(module, 'last_xs_b') and module.last_xs_b is not None:
                xs_b_np = module.last_xs_b.to(export_dtype).detach().cpu().numpy()
                xs_b_np.tofile(f"{save_folder}/{name}.xs_b.{dtype_text}.bin")

            if hasattr(module, 'last_out_z_b') and module.last_out_z_b is not None:
                out_z_b_np = module.last_out_z_b.to(export_dtype).detach().cpu().numpy()
                out_z_b_np.tofile(f"{save_folder}/{name}.out_z_b.{dtype_text}.bin")
                
        # Save output tensor
        if output is not None:
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            
            if isinstance(module, Block):
                if isinstance(output, tuple):
                    last_hidden = output[0]
                    last_residual = output[1] if len(output) > 1 else None
                else:
                    last_hidden = output
                    last_residual = None
                final_norm_capture['hidden'] = last_hidden.detach().cpu()
                final_norm_capture['residual'] = last_residual.detach().cpu() if last_residual is not None else None

            output_np = output_tensor.to(export_dtype).detach().cpu().numpy()

            if isinstance(module, QuantLinear) and 'in_proj' in name:
                output_np_x = output_np[:, :, :384]
                output_np_z = output_np[:, :, 384:]
                output_np_z_silu = output_np_z * (1 / (1 + np.exp(-output_np_z)))
                output_np_x.tofile(f"{save_folder}/{name}.output.x.{dtype_text}.bin")
                output_np_z_silu.tofile(f"{save_folder}/{name}.output.z_silu.{dtype_text}.bin")
            elif isinstance(module, QuantConv1d) and output_np.ndim == 3:
                output_np_tmp = output_np[:, :, :197]
                output_np = output_np_tmp.transpose(0, 2, 1)
                output_np.tofile(f"{save_folder}/{name}.output.{dtype_text}.bin")
            else:
                output_np.tofile(f"{save_folder}/{name}.output.{dtype_text}.bin")
    
    # Register hooks only for target layer types
    for name, module in model.named_modules():
        if isinstance(module, (Block, QuantLinear, QuantConv1d, MambaModule)):
            # Include both forward and backward (_b) modules
            if isinstance(module, Block):
                name = f"{name}.norm"
                
            hook = module.register_forward_hook(
                functools.partial(hook_fn, name=name)
            )
            hooks.append(hook)
    
    # Run inference on one batch to capture inputs and outputs
    for batch_idx, (input_tensor, _) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor[0].unsqueeze(0)
        input_tensor = input_tensor.to(export_dtype)
        
        print(f"Processing batch {batch_idx} with shape: {input_tensor.shape}")
        
        # Save the input image
        img_np = input_tensor.cpu().numpy().squeeze(0)
        img_np.tofile(f"{image_folder}/image.{dtype_text}.bin")
        print(f"Saved input image to {image_folder}/image.{dtype_text}.bin - shape: {img_np.shape}")
        
        with torch.no_grad():
            output = model(input_tensor)
            
            # Save final model output
            final_output_np = output.to(export_dtype).detach().cpu().numpy()
            padded_final_output_np = np.zeros((1, 1008), dtype=np_dtype)
            padded_final_output_np[0, :final_output_np.shape[1]] = final_output_np
            padded_final_output_np.tofile(f"{save_folder}/final_output.{dtype_text}.bin")
            
            # Save final norm input and output
            device_for_norm = model.norm_f.weight.device
            hidden = final_norm_capture['hidden'].to(device_for_norm)
            residual = final_norm_capture['residual'].to(device_for_norm) if final_norm_capture['residual'] is not None else None
            real_residual = hidden+residual
            real_residual.to(export_dtype).detach().cpu().numpy().tofile(f"{save_folder}/last_residual.{dtype_text}.bin")
            cls_token = real_residual[:, 98, :]
            cls_token.to(export_dtype).detach().cpu().numpy().tofile(f"{save_folder}/norm_f.input.{dtype_text}.bin")
            final_norm_out = models_mamba.rms_norm_fn(
                hidden,
                model.norm_f.weight,
                model.norm_f.bias,
                eps=model.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=model.residual_in_fp32,
            )
            final_norm = final_norm_out[:, 98, :]
            final_norm.to(export_dtype).detach().cpu().numpy().tofile(f"{save_folder}/norm_f.output.{dtype_text}.bin")
        
        break  # Only process first batch
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"Intermediate inputs and outputs saved in {save_folder}")
    return save_folder, image_folder

def export_parameters(model, model_config, save_dir_base, num_classes=1000, export_dtype=torch.float32, reorder_to_blocks=False, block_size=16, pack_int4=False):
    """
    Export model parameters to binary files.
    
    Args:
        model: The model to export
        model_config: Model configuration dictionary
        save_dir_base: Base directory to save exported files
        num_classes: Number of output classes
        export_dtype: Data type for export (torch.float32 or torch.float16)
        reorder_to_blocks: Whether to reorder linear weights to block format for HLS
        block_size: Block size for weight reordering (default: 16)
    """
    dtype_text = "float32" if export_dtype == torch.float32 else "float16"
    np_dtype = np.float32 if export_dtype == torch.float32 else np.float16
    save_folder = os.path.join(save_dir_base, f"bin_{dtype_text}_block")

    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    state_dict = model.state_dict()
    print(f"Exporting model parameters to {save_folder}...")
    
    if reorder_to_blocks:
        print(f"Block reordering enabled: Linear weights will be reordered to {block_size}x{block_size} blocks for HLS implementation")
    else:
        print("Block reordering disabled: Linear weights will be saved in standard row-major format")
    
    # Helper to save a tensor and count parameters
    def save_tensor(tensor, name, is_quantized_weight=False, compared_shape=None, split_sign_magnitude=False, reorder_to_blocks=False, out_dim=None, in_dim=None, block_size=16, pack_int4=False):
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = tensor
            
        if compared_shape is not None:
            if tensor_np.shape != compared_shape:
                raise ValueError(f"Shape mismatch for {name}: expected {compared_shape}, got {tensor_np.shape}")
                return
        
        # Reorder to blocks if requested (for linear weights)
        if reorder_to_blocks and out_dim is not None and in_dim is not None:
            if tensor_np.shape != (out_dim, in_dim):
                print(f"Warning: Expected shape ({out_dim}, {in_dim}) for {name}, got {tensor_np.shape}")
                # Use actual shape if different
                actual_out_dim, actual_in_dim = tensor_np.shape
            else:
                actual_out_dim, actual_in_dim = out_dim, in_dim
            
            # Store original for verification
            original_tensor = tensor_np.copy()
            
            tensor_np = reorder_weights_to_blocks(tensor_np, actual_out_dim, actual_in_dim, block_size)
            print(f"  Reordered {name} to blocks with shape {tensor_np.shape}")
        
        if is_quantized_weight and split_sign_magnitude:
            # Flatten the tensor to 1D for packing (whether it's 2D or 3D after reordering)
            tensor_flat = tensor_np.flatten()
            tensor_int8 = tensor_flat.astype(np.int8)

            # INT4 packing: Extract sign and magnitude, then pack
            # Extract sign bit (bit 4, which is the MSB of 5-bit value)
            sign_bits = ((tensor_int8 & 0x10) >> 4).astype(np.uint8)
            # Extract magnitude (bits 0-3)
            magnitude_bits = (tensor_int8 & 0x0F).astype(np.uint8)
            
            if pack_int4:
                # Check if total number of sign bits is a multiple of 8
                if sign_bits.size % 8 != 0:
                    raise ValueError(f"Total number of sign bits ({sign_bits.size}) must be a multiple of 8 for {name}")
                
                # Pack 4-bit magnitudes into bytes (2 magnitudes per byte)
                magnitude_packed = np.zeros((magnitude_bits.size + 1) // 2, dtype=np.uint8)
                for i in range(0, magnitude_bits.size, 2):
                    if i + 1 < magnitude_bits.size:
                        # Pack two 4-bit values into one byte (first in upper bits, second in lower bits)
                        magnitude_packed[i // 2] = ((magnitude_bits[i] & 0x0F) << 4) | (magnitude_bits[i + 1] & 0x0F)
                    else:
                        # Last value (odd number of elements)
                        magnitude_packed[i // 2] = (magnitude_bits[i] & 0x0F) << 4
                
                # Pack 1-bit signs into bytes (8 signs per byte)
                sign_packed = np.zeros((sign_bits.size + 7) // 8, dtype=np.uint8)
                for i in range(0, sign_bits.size, 8):
                    packed_byte = 0
                    for j in range(8):
                        if i + j < sign_bits.size:
                            packed_byte |= (sign_bits[i + j] & 0x01) << j
                    sign_packed[i // 8] = packed_byte
                
                # Check 256-bit alignment (32 bytes)
                magnitude_bytes = magnitude_packed.size
                sign_bytes = sign_packed.size

                assert magnitude_bytes % 32 == 0, f"magnitude_bytes % 32 != 0: {magnitude_bytes} % 32 != 0"
                assert sign_bytes % 32 == 0, f"sign_bytes % 32 != 0: {sign_bytes} % 32 != 0"
                
                # Save packed sign and magnitude
                sign_path = os.path.join(save_folder, f"{name}.sign.bin")
                magnitude_path = os.path.join(save_folder, f"{name}.magnitude.bin")
                sign_packed.tofile(sign_path)
                magnitude_packed.tofile(magnitude_path)
                print(f"  Saved {name}.magnitude as packed 4-bit with shape {magnitude_packed.shape} ({magnitude_packed.nbytes} bytes)")
                print(f"  Saved {name}.sign as packed 1-bit with shape {sign_packed.shape} ({sign_packed.nbytes} bytes)")
            
            else:
                # Save both sign and magnitude as uint8
                sign_path = os.path.join(save_folder, f"{name}.sign.bin")
                magnitude_path = os.path.join(save_folder, f"{name}.magnitude.bin")
                sign_bits.tofile(sign_path)
                magnitude_bits.tofile(magnitude_path)
                print(f"  Saved {name}.magnitude as uint8 with shape {magnitude_bits.shape} ({magnitude_bits.nbytes} bytes)")
                print(f"  Saved {name}.sign as uint8 with shape {sign_bits.shape} ({sign_bits.nbytes} bytes)")

        elif is_quantized_weight and pack_int4:
            # Flatten the tensor to 1D for packing
            tensor_flat = tensor_np.flatten()
            tensor_int8 = tensor_flat.astype(np.int8)
            
            # Pack two 4-bit values into one byte
            # The weights are 4-bit (bits 0-3), where bit 3 is sign and 0-2 is magnitude
            packed_weights = np.zeros((tensor_int8.size + 1) // 2, dtype=np.uint8)
            for i in range(0, tensor_int8.size, 2):
                if i + 1 < tensor_int8.size:
                    # Pack two 4-bit values into one byte (first in upper bits, second in lower bits)
                    packed_weights[i // 2] = ((tensor_int8[i] & 0x0F) << 4) | (tensor_int8[i + 1] & 0x0F)
                else:
                    # Last value (odd number of elements)
                    packed_weights[i // 2] = (tensor_int8[i] & 0x0F) << 4
            
            # Check 256-bit alignment (32 bytes)
            packed_bytes = packed_weights.size
            if packed_bytes % 32 != 0:
                print(f"  Warning: {name} is {packed_bytes} bytes, not aligned to 256 bits (32 bytes)")
                # Pad to 256-bit boundary
                padding_needed = 32 - (packed_bytes % 32)
                packed_weights = np.pad(packed_weights, (0, padding_needed), mode='constant')
                print(f"  Padded {name} to {packed_weights.size} bytes for 256-bit alignment")
            
            # Save packed weights
            int4_path = os.path.join(save_folder, f"{name}.int4.bin")
            packed_weights.tofile(int4_path)
            print(f"  Saved {name} as packed 4-bit weights with shape {packed_weights.shape} ({packed_weights.nbytes} bytes)")
            
        elif is_quantized_weight:
            int8_path = os.path.join(save_folder, f"{name}.int8.bin")
            tensor_np.astype(np.int8).tofile(int8_path)
            print(f"  Saved {name} as int8 weights with shape {tensor_np.shape} ({tensor_np.astype(np.int8).nbytes} bytes)")
            
        else:
            data_path = os.path.join(save_folder, f"{name}.{dtype_text}.bin")
            export_arr = tensor_np.astype(np_dtype)
            export_arr.tofile(data_path)
            print(f"  Saved {name}.{dtype_text}.bin with shape {export_arr.shape} ({export_arr.nbytes} bytes)")

    print("----- Embedding -----")
    save_tensor(state_dict["cls_token"], "cls_token", compared_shape=(1, 1, model_config['d_model']))
    save_tensor(state_dict["pos_embed"], "pos_embed", compared_shape=(1, model_config['seq_len'], model_config['d_model']))
    save_tensor(state_dict["patch_embed.proj.weight"], "patch_embed.proj.weight", compared_shape=(model_config['d_model'], model_config['n_channels'], model_config['patch_size'], model_config['patch_size']))
    save_tensor(state_dict["patch_embed.proj.bias"], "patch_embed.proj.bias", compared_shape=(model_config['d_model'],))

    for i in range(model_config['depth']):
        layer_prefix = f"layers.{i}"
        print(f"----- {layer_prefix} -----")

        # Norm
        norm_weight = state_dict[f"{layer_prefix}.norm.weight"]
        save_tensor(norm_weight, f"{layer_prefix}.norm.weight", compared_shape=(model_config['d_model'],))

        # Mixer parameters
        mixer_prefix = f"{layer_prefix}.mixer"
        
        # Get the actual mixer module to check if it's quantized
        mixer_module = model.layers[i].mixer
        
        # Define the parameter names that might exist in the mixer
        param_names = [
            "in_proj", "conv1d", "conv1d_b", 
            "x_proj", "x_proj_b", 
            "dt_proj", "dt_proj_b", 
            "out_proj"
        ]

        for param_name in param_names:
            # Check if this parameter exists in the mixer
            if hasattr(mixer_module, param_name):
                param_module = getattr(mixer_module, param_name)
                base_param_name = f"{mixer_prefix}.{param_name}"
                
                # Handle quantized parameters
                if isinstance(param_module, (QuantConv1d,)):
                    # Save quantized weight (as int8) and its scale
                    if hasattr(param_module, 'weight_quant_pot'):
                        weight_quant_pot = rearrange(param_module.weight_quant_pot, 'd 1 k -> d k')
                        print(f"Quantized weight shape for {base_param_name}.weight: {weight_quant_pot.shape}")
                        # For conv1d, we don't need to reorder to blocks
                        out_dim, in_dim = weight_quant_pot.shape
                        save_tensor(weight_quant_pot, f"{base_param_name}.weight", 
                                  is_quantized_weight=True, split_sign_magnitude=True,
                                  reorder_to_blocks=False, out_dim=out_dim, in_dim=in_dim, pack_int4=True)
                    else:
                        raise ValueError(f"Quantized weight not found in {base_param_name}.weight")
                    
                    # Save weight scale
                    if hasattr(param_module, 'weight_scales') and hasattr(param_module, 'proj_scale'):
                        combined_scale = param_module.weight_scales * param_module.proj_scale
                        save_tensor(combined_scale, f"{base_param_name}.weight_scale")
                    else:
                        raise ValueError(f"Weight scales not found in {base_param_name}.weight_scale")
                    
                    # Save bias if exists
                    if hasattr(param_module, 'bias') and param_module.bias is not None:
                        save_tensor(param_module.bias, f"{base_param_name}.bias")
                        
                elif isinstance(param_module, (QuantLinear,)) and 'in_proj' in param_name:
                    # Handle in_proj: split into x and z parts
                    if hasattr(param_module, 'weight_quant_pot'):
                        out_dim, in_dim = param_module.weight_quant_pot.shape
                        d_inner = out_dim // 2
                        
                        weight_x = param_module.weight_quant_pot[:d_inner, :]
                        weight_z = param_module.weight_quant_pot[d_inner:, :]
                        
                        save_tensor(weight_x, f"{base_param_name}.weight.x", 
                                  is_quantized_weight=True, split_sign_magnitude=False,
                                  reorder_to_blocks=reorder_to_blocks, out_dim=d_inner, in_dim=in_dim, block_size=block_size, pack_int4=pack_int4)
                        save_tensor(weight_z, f"{base_param_name}.weight.z", 
                                  is_quantized_weight=True, split_sign_magnitude=False,
                                  reorder_to_blocks=reorder_to_blocks, out_dim=d_inner, in_dim=in_dim, block_size=block_size, pack_int4=pack_int4)
                    else:
                        raise ValueError(f"Quantized weight not found in {base_param_name}.weight")
                    
                    # Save weight scale: split into x and z parts
                    if hasattr(param_module, 'weight_scales') and hasattr(param_module, 'proj_scale'):
                        combined_scale = (param_module.weight_scales * param_module.proj_scale)
                        if combined_scale.dim() == 2:
                            # Transpose from [out_dim, num_in_blocks] to [num_in_blocks, out_dim]
                            combined_scale = combined_scale.t()
                            d_inner = combined_scale.shape[1] // 2
                            scale_x = combined_scale[:, :d_inner].flatten()
                            scale_z = combined_scale[:, d_inner:].flatten()
                        else:
                            d_inner = combined_scale.shape[0] // 2
                            scale_x = combined_scale[:d_inner].flatten()
                            scale_z = combined_scale[d_inner:].flatten()
                        
                        save_tensor(scale_x, f"{base_param_name}.weight_scale.x")
                        save_tensor(scale_z, f"{base_param_name}.weight_scale.z")
                    else:
                        raise ValueError(f"Weight scales not found in {base_param_name}.weight_scale")
                    
                    # Save bias if exists: split into x and z parts
                    if hasattr(param_module, 'bias') and param_module.bias is not None:
                        d_inner = param_module.bias.shape[0] // 2
                        bias_x = param_module.bias[:d_inner]
                        bias_z = param_module.bias[d_inner:]
                        save_tensor(bias_x, f"{base_param_name}.bias.x")
                        save_tensor(bias_z, f"{base_param_name}.bias.z")
                        
                    # Save smooth scale if exists
                    if hasattr(param_module, 'smooth_scales') and param_module.smooth_scales is not None:
                        save_tensor(param_module.smooth_scales, f"{base_param_name}.smooth_scales")

                elif isinstance(param_module, (QuantLinear,)) and 'x_proj' not in param_name and 'dt_proj' not in param_name and 'in_proj' not in param_name:
                    # Save quantized weight (as int8) and its scale
                    if hasattr(param_module, 'weight_quant_pot'):
                        # Get dimensions for block reordering
                        out_dim, in_dim = param_module.weight_quant_pot.shape
                        save_tensor(param_module.weight_quant_pot, f"{base_param_name}.weight", 
                                  is_quantized_weight=True, split_sign_magnitude=False,
                                  reorder_to_blocks=reorder_to_blocks, out_dim=out_dim, in_dim=in_dim, block_size=block_size, pack_int4=pack_int4)
                    else:
                        raise ValueError(f"Quantized weight not found in {base_param_name}.weight")
                    
                    # Save weight scale
                    if hasattr(param_module, 'weight_scales') and hasattr(param_module, 'proj_scale'):
                        combined_scale = (param_module.weight_scales * param_module.proj_scale)
                        if combined_scale.dim() == 2:
                            # Transpose from [out_dim, num_in_blocks] to [num_in_blocks, out_dim]
                            combined_scale = combined_scale.t()
                        
                        save_tensor(combined_scale.flatten(), f"{base_param_name}.weight_scale")
                    else:
                        raise ValueError(f"Weight scales not found in {base_param_name}.weight_scale")
                    
                    # Save bias if exists
                    if hasattr(param_module, 'bias') and param_module.bias is not None:
                        save_tensor(param_module.bias, f"{base_param_name}.bias")
                        
                    # Save smooth scale if exists
                    if hasattr(param_module, 'smooth_scales') and param_module.smooth_scales is not None:
                        save_tensor(param_module.smooth_scales, f"{base_param_name}.smooth_scales")
                
                elif isinstance(param_module, (QuantLinear,)) and 'dt_proj' in param_name:
                    # Handle dt_proj: pad from dt_rank (12) to 16 dimensions
                    if hasattr(param_module, 'weight_quant_pot'):
                        dt_weight = param_module.weight_quant_pot  # Shape: [d_inner, dt_rank]
                        # Pad with zeros from dt_rank to 16
                        padded_dt_weight = torch.zeros(model_config['d_inner'], 16, dtype=dt_weight.dtype, device=dt_weight.device)
                        padded_dt_weight[:, :model_config['dt_rank']] = dt_weight
                        save_tensor(padded_dt_weight, f"{base_param_name}.weight", 
                                  is_quantized_weight=True, compared_shape=(model_config['d_inner'], 16), 
                                  split_sign_magnitude=False,
                                  reorder_to_blocks=reorder_to_blocks, out_dim=model_config['d_inner'], in_dim=16, block_size=block_size, pack_int4=pack_int4)
                    else:
                        raise ValueError(f"Quantized weight not found in {base_param_name}.weight")
                    
                    # Handle dt_proj weight scale: pad from dt_rank to 16
                    if hasattr(param_module, 'weight_scales') and hasattr(param_module, 'proj_scale'):
                        dt_scale = (param_module.weight_scales * param_module.proj_scale)
                        if dt_scale.dim() == 2:
                            # Transpose from [out_dim, num_in_blocks] to [num_in_blocks, out_dim]
                            dt_scale = dt_scale.t()
                            num_in_blocks = dt_scale.shape[0]
                            save_tensor(dt_scale.flatten(), f"{base_param_name}.weight_scale", compared_shape=(num_in_blocks * model_config['d_inner'],))
                        else:
                            save_tensor(dt_scale.flatten(), f"{base_param_name}.weight_scale", compared_shape=(model_config['d_inner'],))
                    else:
                        raise ValueError(f"Weight scales not found in {base_param_name}.weight_scale")
                    
                    # Handle dt_proj bias: pad from dt_rank to 16 is not needed (bias is d_inner)
                    if hasattr(param_module, 'bias') and param_module.bias is not None:
                        save_tensor(param_module.bias, f"{base_param_name}.bias", compared_shape=(model_config['d_inner'],))
                
                elif isinstance(param_module, (QuantLinear,)) and 'x_proj' in param_name:
                    # Handle x_proj: pad dt part to 16 and merge with BC part
                    if hasattr(param_module, 'weight_quant_pot'):
                        # Original x_proj weight shape: [dt_rank + 2*d_state, d_inner]
                        original_weight = param_module.weight_quant_pot
                        
                        # Split into dt, B and C parts
                        dt_part = original_weight[:model_config['dt_rank'], :]  # Shape: [dt_rank, d_inner]
                        b_part = original_weight[model_config['dt_rank']:model_config['dt_rank']+model_config['d_state'], :]   # Shape: [d_state, d_inner]
                        c_part = original_weight[model_config['dt_rank']+model_config['d_state']:, :]   # Shape: [d_state, d_inner]
                        
                        # Pad dt part from dt_rank to 16
                        padded_dt_part = torch.zeros(16, model_config['d_inner'], dtype=dt_part.dtype, device=dt_part.device)
                        padded_dt_part[:model_config['dt_rank'], :] = dt_part
                        
                        save_tensor(padded_dt_part, f"{base_param_name}.weight.dt", 
                                  is_quantized_weight=True, compared_shape=(16, model_config['d_inner']), 
                                  split_sign_magnitude=False,
                                  reorder_to_blocks=reorder_to_blocks, out_dim=16, in_dim=model_config['d_inner'], block_size=block_size, pack_int4=pack_int4)
                        save_tensor(b_part, f"{base_param_name}.weight.b", 
                                  is_quantized_weight=True, compared_shape=(model_config['d_state'], model_config['d_inner']), 
                                  split_sign_magnitude=False,
                                  reorder_to_blocks=reorder_to_blocks, out_dim=model_config['d_state'], in_dim=model_config['d_inner'], block_size=block_size, pack_int4=pack_int4)
                        save_tensor(c_part, f"{base_param_name}.weight.c", 
                                  is_quantized_weight=True, compared_shape=(model_config['d_state'], model_config['d_inner']), 
                                  split_sign_magnitude=False,
                                  reorder_to_blocks=reorder_to_blocks, out_dim=model_config['d_state'], in_dim=model_config['d_inner'], block_size=block_size, pack_int4=pack_int4)
                    else:
                        raise ValueError(f"Quantized weight not found in {base_param_name}.weight")
                    
                    # Handle weight scale: pad dt part and merge with BC part
                    if hasattr(param_module, 'weight_scales') and hasattr(param_module, 'proj_scale'):
                        combined_scale = (param_module.weight_scales * param_module.proj_scale)
                        
                        if combined_scale.dim() == 2:
                            # Transpose from [out_dim, num_in_blocks] to [num_in_blocks, out_dim]
                            combined_scale = combined_scale.t()
                            num_in_blocks = combined_scale.shape[0]
                            
                            # Split into dt, B and C parts along the output dimension (now axis 1)
                            dt_scale = combined_scale[:, :model_config['dt_rank']]
                            b_scale = combined_scale[:, model_config['dt_rank']:model_config['dt_rank']+model_config['d_state']]
                            c_scale = combined_scale[:, model_config['dt_rank']+model_config['d_state']:]
                            
                            # Pad dt scale from dt_rank to 16 along the output dimension
                            padded_dt_scale = torch.zeros(num_in_blocks, 16, dtype=dt_scale.dtype, device=dt_scale.device)
                            padded_dt_scale[:, :model_config['dt_rank']] = dt_scale
                            
                            save_tensor(padded_dt_scale.flatten(), f"{base_param_name}.weight_scale.dt", compared_shape=(16 * num_in_blocks,))
                            save_tensor(b_scale.flatten(), f"{base_param_name}.weight_scale.b", compared_shape=(model_config['d_state'] * num_in_blocks,))
                            save_tensor(c_scale.flatten(), f"{base_param_name}.weight_scale.c", compared_shape=(model_config['d_state'] * num_in_blocks,))
                        else:
                            # per_channel: (out_features, 1) or (out_features,)
                            combined_scale = combined_scale.flatten()
                            dt_scale = combined_scale[:model_config['dt_rank']]
                            b_scale = combined_scale[model_config['dt_rank']:model_config['dt_rank']+model_config['d_state']]
                            c_scale = combined_scale[model_config['dt_rank']+model_config['d_state']:]
                            
                            padded_dt_scale = torch.zeros(16, dtype=dt_scale.dtype, device=dt_scale.device)
                            padded_dt_scale[:model_config['dt_rank']] = dt_scale
                            
                            save_tensor(padded_dt_scale, f"{base_param_name}.weight_scale.dt", compared_shape=(16,))
                            save_tensor(b_scale, f"{base_param_name}.weight_scale.b", compared_shape=(model_config['d_state'],))
                            save_tensor(c_scale, f"{base_param_name}.weight_scale.c", compared_shape=(model_config['d_state'],))
                    else:
                        raise ValueError(f"Weight scales not found in {base_param_name}.weight_scale")

                    # Save smooth scale if exists (no change needed as it's per input dimension)
                    if hasattr(param_module, 'smooth_scales') and param_module.smooth_scales is not None:
                        save_tensor(param_module.smooth_scales, f"{base_param_name}.smooth_scales", compared_shape=(model_config['d_inner'],))
                
                else:
                    raise ValueError(f"Unsupported mixer parameter type for {base_param_name}")
        
        # A and D parameters (these are typically not quantized)
        A_log_key = f"{mixer_prefix}.A_log"
        A_b_log_key = f"{mixer_prefix}.A_b_log"
        D_key = f"{mixer_prefix}.D"
        D_b_key = f"{mixer_prefix}.D_b"
        
        if A_log_key in state_dict:
            A = -torch.exp(state_dict[A_log_key])
            save_tensor(A, f"{mixer_prefix}.A")
        if A_b_log_key in state_dict:
            A_b = -torch.exp(state_dict[A_b_log_key])
            save_tensor(A_b, f"{mixer_prefix}.A_b")
        
        if D_key in state_dict:
            save_tensor(state_dict[D_key], f"{mixer_prefix}.D")
        if D_b_key in state_dict:
            save_tensor(state_dict[D_b_key], f"{mixer_prefix}.D_b")

    print("----- Output Head -----")
    # Final Norm
    if "norm_f.weight" in state_dict:
        final_norm_weight = state_dict["norm_f.weight"]
        save_tensor(final_norm_weight, "norm_f.weight")

    # Head
    head_module = model.head
    if isinstance(head_module, QuantLinear):
        # Handle quantized head
        padded_head_dim0 = (num_classes + 15) // 16 * 16
        
        # Save quantized weight
        if hasattr(head_module, 'weight_quant_pot'):
            padded_weight = torch.zeros(padded_head_dim0, head_module.weight_quant_pot.shape[1], dtype=torch.int8)
            padded_weight[:num_classes, :] = head_module.weight_quant_pot
            save_tensor(padded_weight, "head.weight", 
                      is_quantized_weight=True, split_sign_magnitude=False,
                      reorder_to_blocks=reorder_to_blocks, out_dim=padded_head_dim0, in_dim=head_module.weight_quant_pot.shape[1], block_size=block_size, pack_int4=pack_int4)
        else:
            raise ValueError("Quantized weight not found in head module.")
        
        # Save weight scale
        if hasattr(head_module, 'weight_scales'):
            combined_scale = (head_module.weight_scales * head_module.proj_scale)
            if combined_scale.dim() == 2:
                # Transpose from [out_dim, num_in_blocks] to [num_in_blocks, out_dim]
                combined_scale = combined_scale.t()
                num_in_blocks = combined_scale.shape[0]
                padded_scale = torch.zeros(num_in_blocks, padded_head_dim0, dtype=export_dtype)
                padded_scale[:, :num_classes] = combined_scale
            else:
                padded_scale = torch.zeros(padded_head_dim0, dtype=export_dtype)
                padded_scale[:num_classes] = combined_scale.flatten()
            
            save_tensor(padded_scale.flatten(), "head.weight_scale")
        else:
            raise ValueError("Weight scales not found in head module.")
        
        # Save bias
        if hasattr(head_module, 'bias') and head_module.bias is not None:
            padded_bias = torch.zeros(padded_head_dim0, dtype=export_dtype)
            padded_bias[:num_classes] = head_module.bias
            save_tensor(padded_bias, "head.bias")
        else:
            raise ValueError("Bias not found in head module.")
    
    else:
        raise ValueError("Head is not a QuantLinear module, cannot export quantized parameters.")

    print(f"Export finished successfully. Parameters saved in {save_folder}")
    return

def get_args_parser():
    parser = argparse.ArgumentParser('Vim Model Exporter', add_help=False)
    
    # Essential model parameters
    parser.add_argument('--model', default='vim_tiny_pretrain', type=str, metavar='MODEL',
                        help='Name of model to export')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    # Dataset parameters (minimal - needed for num_classes)
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--batch-size', default=1, type=int, help='batch size for data loading')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    
    # Essential system parameters
    parser.add_argument('--output_dir', default='output', help='path where to save exported files')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    # Memory and performance
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    
    # AMP settings
    parser.add_argument('--if_amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp', help='')
    parser.set_defaults(if_amp=False)
    
    # Debug mode
    parser.add_argument('--debug', action='store_true', help='enable debug mode with limited data')
    parser.set_defaults(debug=False)
    
    # Quantization parameters
    parser.add_argument('--smooth', action='store_true', help='Use smooth quantization')
    parser.add_argument('--act_scales', default='', help='path for act_scale checkpoint')
    
    # Export specific parameters
    parser.add_argument('--export', action='store_true', help='Export model parameters to binary files')
    parser.add_argument('--export-dtype', default='float32', choices=['float32', 'float16'], 
                        help='Export data type (default: float32)')
    parser.add_argument('--reorder-to-blocks', action='store_true', default=True,
                        help='Reorder linear weights to block format for HLS (default: True)')
    parser.add_argument('--no-reorder-to-blocks', action='store_false', dest='reorder_to_blocks',
                        help='Disable block reordering for linear weights')
    parser.add_argument('--block-size', type=int, default=16,
                        help='Block size for weight reordering (default: 16)')
    parser.add_argument('--pack-int4', action='store_true', default=True,
                        help='Pack quantized weights as INT4 (2 weights per byte) for memory efficiency (default: True)')
    parser.add_argument('--no-pack-int4', action='store_false', dest='pack_int4',
                        help='Store quantized weights as INT8 without packing')
    
    # Optional evaluation
    parser.add_argument('--eval', action='store_true', help='Perform evaluation before export')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    
    # Save intermediate activations for debugging
    parser.add_argument('--save-intermediates', action='store_true', 
                        help='Save intermediate activations for debugging')
    parser.add_argument('--manual_scan', action='store_true', default=True, 
                        help='Use manual scan in MambaModule forward pass')
    
    # Layer limiting for testing/debugging
    parser.add_argument('--max-layers', type=int, default=1, help='Limit the number of layers to process. None = all layers')
    
    return parser

def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    ## Data Loader
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
            
    ## Debugging
    if args.debug:
        if args.eval or args.export:
            debug_size_val = min(len(dataset_val), 10)          # Use only 10 samples or fewer for validation
            dataset_val = torch.utils.data.Subset(dataset_val, range(debug_size_val))
        else:
            debug_size_train = min(len(dataset_train), 500)     # Use only 500 samples or fewer for training
            debug_size_val = min(len(dataset_val), 200)         # Use only 200 samples or fewer for validation
            dataset_train = torch.utils.data.Subset(dataset_train, range(debug_size_train))
            dataset_val = torch.utils.data.Subset(dataset_val, range(debug_size_val))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size * 20),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print("Train loader size: ", len(data_loader_train))
    print("Validation loader size: ", len(data_loader_val))
    
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        manual_scan=args.manual_scan,
        max_layers=args.max_layers
    )
    if args.max_layers is not None:
        print(f"WARNING: Model will only process {args.max_layers} layer(s) out of {len(model.layers)} total layers")
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print(model)
    
    ### Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model']
        
        ## Smooth the layers and quantize the model
        model.load_state_dict(state_dict, strict=True)
        if args.smooth:
            scale_file = f"{args.model}_smoothing.pt"
            if '.pt' not in args.act_scales:
                act_scales_path = f"{args.act_scales}/{scale_file}"
            else:
                act_scales_path = args.act_scales
            act_scales = torch.load(act_scales_path, map_location='cpu')
            smooth_vim_model(model, act_scales, dtype=torch.float32, alpha=0.5)
            quantize_vim_model(model, weight_bits=4, act_bits=8, 
                                linear_act_per_token=True, conv_act_per_channel=False,
                                power=True, additive=True,
                                per_block=True, block_size=32,
                                quantize_head=True, ptq=True, static_quant=False)
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('number of params:', n_parameters)
    
    ## Evaluation
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
    ## Export model parameters
    if args.export:
        if not args.resume:
            print("!!! Warning: No checkpoint loaded. Exporting untrained model parameters.")
        
        # Set export data type
        export_dtype = torch.float32 if args.export_dtype == 'float32' else torch.float16
        
        model_config = {
            'depth': 24,
            'image_size': 224,
            'n_channels': 3,
            'patch_size': 16,
            'num_patches': (224 // 16) * (224 // 16),
            'seq_len': (224 // 16) * (224 // 16) + 1,  # +1 for cls token
            'd_model': 192,
            'd_inner': 192*2,
            'd_state': 16,
            'dt_rank': 12,  # Actual dt_rank used in model
        }
        
        # Export parameters
        export_parameters(
            model,
            model_config,
            args.output_dir,
            num_classes=args.nb_classes,
            export_dtype=export_dtype,
            reorder_to_blocks=args.reorder_to_blocks,
            block_size=args.block_size,
            pack_int4=args.pack_int4
        )

    ## Save intermediate outputs for debugging
    if args.save_intermediates:
        if not args.resume:
            print("Warning: Saving intermediates from untrained model.")
        
        print("Saving intermediate outputs...")
        export_dtype = torch.float32 if args.export_dtype == 'float32' else torch.float16
        save_intermediate_outputs(model, data_loader_val, args.output_dir, device=device, export_dtype=export_dtype)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vim Model Exporter', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
