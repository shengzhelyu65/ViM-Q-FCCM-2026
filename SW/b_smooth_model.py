import torch
import torch.nn as nn

from models_mamba import Block
from models_mamba import RMSNorm

@torch.no_grad()
def smooth_fc(fc, act_scales, dtype=torch.float32, alpha=0.5):
    assert isinstance(fc, nn.Linear)
    
    device = fc.weight.device
    act_scales = act_scales.to(device=device, dtype=dtype) # [in_features]
    
    weight_scales = fc.weight.abs().max(dim=0, keepdim=True)[0] # [out_features, in_features] -> [1, in_features]
    weight_scales = weight_scales[0].clamp(min=1e-5) # [in_features]
    
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    scales_inv = torch.div(1, scales).to(device=device, dtype=dtype)
    fc.register_parameter(
        name="smooth_scales",
        param=torch.nn.Parameter(scales_inv, requires_grad=False),
    )
    
    fc.weight.mul_(scales.reshape(1, -1)) # [out_features, in_features] * [1, in_features] = [out_features, in_features]

@torch.no_grad()
def smooth_in_proj(norm, in_proj, act_scales, dtype=torch.float32, alpha=0.5):
    assert isinstance(norm, RMSNorm)
    assert isinstance(in_proj, nn.Linear)
    
    device = in_proj.weight.device
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    weight_scales = in_proj.weight.abs().max(dim=0, keepdim=True)[0]
    weight_scales = weight_scales[0].clamp(min=1e-5)
    
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    norm.weight.div_(scales)
    assert norm.bias is None, "RMSNorm bias should be None"
    
    in_proj.weight.mul_(scales.reshape(1, -1))
    
@torch.no_grad()
def smooth_both_conv(in_proj, conv, conv_b, act_scales, act_scales_b, dtype=torch.float32, alpha=0.5, scale_merge='max'):
    assert isinstance(in_proj, nn.Linear)
    assert isinstance(conv, nn.Conv1d)
    
    device = in_proj.weight.device
    act_scales = act_scales.to(device=device, dtype=dtype)
    act_scales_b = act_scales_b.to(device=device, dtype=dtype)
    
    conv_weight = conv.weight.squeeze(1)
    weight_scales = conv_weight.abs().max(dim=1, keepdim=True)[0]
    weight_scales = weight_scales[:, 0].clamp(min=1e-5)
    
    conv_b_weight = conv_b.weight.squeeze(1)
    weight_scales_b = conv_b_weight.abs().max(dim=1, keepdim=True)[0]
    weight_scales_b = weight_scales_b[:, 0].clamp(min=1e-5)
    
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    scales_b = (
        (act_scales_b.pow(alpha) / weight_scales_b.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    scales = torch.stack([scales, scales_b])
    if scale_merge == 'mean':
        scales = scales.mean(dim=0)
    elif scale_merge == 'max':
        scales = scales.max(dim=0)[0]
    
    hidden_dim = conv.weight.shape[0]
    
    in_proj.weight[:hidden_dim].div_(scales.reshape(-1, 1))
    
    conv.weight.mul_(scales.reshape(-1, 1, 1))
    conv_b.weight.mul_(scales.reshape(-1, 1, 1))

@torch.no_grad()
def smooth_dt_x_proj(dt_proj, x_proj, dt_act_scales, x_act_scales, dtype=torch.float32, alpha=0.5):
    assert isinstance(dt_proj, nn.Linear)
    assert isinstance(x_proj, nn.Linear)
    
    device = x_proj.weight.device
    x_act_scales = x_act_scales.to(device=device, dtype=dtype)
    x_weight_scales = x_proj.weight.abs().max(dim=0, keepdim=True)[0]
    x_weight_scales = x_weight_scales[0].clamp(min=1e-5)
    x_scales = (
        (x_act_scales.pow(alpha) / x_weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    x_proj.weight.mul_(x_scales.reshape(1, -1))
    x_scales_inv = torch.div(1, x_scales).to(device=device, dtype=dtype)
    x_proj.register_parameter(
        name="smooth_scales",
        param=torch.nn.Parameter(x_scales_inv, requires_grad=False),
    )
    
    dt_act_scales = dt_act_scales.to(device=device, dtype=dtype)
    dt_weight_scales = dt_proj.weight.abs().max(dim=0, keepdim=True)[0]
    dt_weight_scales = dt_weight_scales[0].clamp(min=1e-5)
    dt_scales = (
        (dt_act_scales.pow(alpha) / dt_weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    x_proj.weight[:dt_proj.weight.shape[1], :].div_(dt_scales.reshape(-1, 1))
    dt_proj.weight.mul_(dt_scales.reshape(1, -1))
        
@torch.no_grad()
def smooth_vim_model(model, scales, dtype=torch.float32, alpha=0.5, scale_merge='max'):
    for name, module in model.named_modules():
        if isinstance(module, Block):
            rms_norm = getattr(module, 'norm')
            
            in_proj = getattr(module.mixer, 'in_proj')
            in_proj_scales = scales[f"{name}.mixer.in_proj"]
            
            conv1d = getattr(module.mixer, 'conv1d')
            conv1d_scales = scales[f"{name}.mixer.conv1d"]
            conv1d_b = getattr(module.mixer, 'conv1d_b')
            conv1d_b_scales = scales[f"{name}.mixer.conv1d_b"]
            
            dt_proj = getattr(module.mixer, 'dt_proj')
            dt_proj_scales = scales[f"{name}.mixer.dt_proj"]
            x_proj = getattr(module.mixer, 'x_proj')
            x_proj_scales = scales[f"{name}.mixer.x_proj"]
            
            dt_proj_b = getattr(module.mixer, 'dt_proj_b')
            dt_proj_b_scales = scales[f"{name}.mixer.dt_proj_b"]
            x_proj_b = getattr(module.mixer, 'x_proj_b')
            x_proj_b_scales = scales[f"{name}.mixer.x_proj_b"]
            
            out_proj = getattr(module.mixer, 'out_proj')
            out_proj_scales = scales[f"{name}.mixer.out_proj"]
            
            ## In_proj
            smooth_in_proj(rms_norm, in_proj, in_proj_scales, dtype=dtype, alpha=alpha)
            
            ## Conv1d
            smooth_both_conv(in_proj, conv1d, conv1d_b, conv1d_scales, conv1d_b_scales, dtype=dtype, alpha=alpha, scale_merge=scale_merge)
            
            ## x_proj and dt_proj
            smooth_dt_x_proj(dt_proj, x_proj, dt_proj_scales, x_proj_scales, dtype=dtype, alpha=alpha)
            smooth_dt_x_proj(dt_proj_b, x_proj_b, dt_proj_b_scales, x_proj_b_scales, dtype=dtype, alpha=alpha)
            
            ## Out_proj
            smooth_fc(out_proj, out_proj_scales, dtype=dtype, alpha=alpha)

        elif isinstance(module, nn.Linear) and 'head' in name:
            final_rms_norm = getattr(model, 'norm_f')
            smooth_in_proj(final_rms_norm, module, scales[name], dtype=dtype, alpha=alpha)
            
        else:
            continue