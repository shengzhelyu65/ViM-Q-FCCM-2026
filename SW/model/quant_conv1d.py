import math
import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from model.quant_utils import build_power_value
except ImportError:
    from quant_utils import build_power_value

class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scales, weight_qmax, proj_set, int_weights, power, weight_bits, return_int_weights=False):
        if power:
            w_norm = torch.div(weight, scales.view(-1, 1, 1)).clamp(-1, 1)
            sign = w_norm.sign()
            
            # Find closest positive value
            dist = (proj_set.unsqueeze(1).unsqueeze(2).unsqueeze(3) - w_norm.abs().unsqueeze(0)).abs()
            idx = dist.argmin(dim=0)
            
            q_w = proj_set[idx] * sign
            
            if return_int_weights:
                # Get corresponding integer weights
                q_w_int = int_weights[idx].clone()
                negative_mask = (sign < 0)
                # Set sign bit based on weight_bits
                sign_bit_pos = weight_bits - 1
                q_w_int[negative_mask] |= (1 << sign_bit_pos)
                return q_w, q_w_int
            else:
                return q_w
        else:
            w = torch.div(weight, scales.view(-1, 1, 1))
            q_w = torch.round(w)
            assert torch.allclose(q_w, torch.clamp(q_w, min=-weight_qmax-1, max=weight_qmax)), "Quantized weight out of range"
            
            if return_int_weights:
                q_w_int = q_w.long()
                return q_w, q_w_int
            else:
                return q_w

    @staticmethod
    def backward(ctx, grad_output, grad_int_weights=None):
        # STE: Pass gradients through directly to the input (self.weight)
        return grad_output, None, None, None, None, None, None, None, None

class QuantConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        
        weight_bits=5,
        act_bits=8,
        act_per_channel=True,
        power=True,
        additive=True,
        ptq=False,
        static_quant=False,
        layer_name=""
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias
        )
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.act_per_channel = act_per_channel
        self.power = power
        self.additive = additive
        self.ptq = ptq
        self.static_quant = static_quant
        self.layer_name = layer_name
        self.weight_qmax = 2 ** (self.weight_bits - 1) - 1
        self.act_qmax = 2 ** (self.act_bits - 1) - 1
        out_per_group = self.out_channels // self.groups
        self.register_buffer("_out_channel_group_idx", torch.arange(self.out_channels) // out_per_group)
        self.register_buffer("_weight_dequant_scaled_cache", None)
        
    @torch.no_grad()
    def get_weight_scales(self):
        weight_abs_max = self.weight.abs().reshape(self.out_channels, -1).max(dim=1)[0].clamp(min=1e-5)
        weight_scales = weight_abs_max
        weight_scales = weight_scales if self.power else torch.div(weight_scales, self.weight_qmax)
        return weight_scales
    
    @torch.no_grad()
    def from_float(self, float_module):
        self.weight = float_module.weight
        if float_module.bias is not None:
            self.bias = float_module.bias
        else:
            self.bias = None
        
        self.register_buffer("weight_scales", self.get_weight_scales())
        self.act_scales = None
        
        proj_set, proj_scale, int_weights, _, _, _, _ = build_power_value(B=self.weight_bits - 1, additive=self.additive)
        self.register_buffer("proj_set", proj_set.to(self.weight.device))
        self.register_buffer("proj_scale", proj_scale.to(self.weight.device))
        self.register_buffer("int_weights", int_weights.to(self.weight.device))
        
        # Pre-compute quantized weights for PTQ mode
        q_w, q_w_int = QuantizeSTE.apply(
            self.weight, self.weight_scales, self.weight_qmax,
            self.proj_set, self.int_weights,
            self.power, self.weight_bits, True
        )
        self.register_buffer("weight_quant", q_w.to(self.weight.device))
        self.register_buffer("weight_quant_pot", q_w_int.to(self.weight.device))
        self.register_buffer("weight_dequant", (self.weight_quant * self.weight_scales.view(-1, 1, 1)).to(self.weight.device))
        
        if self.ptq:
            del self.weight, self.weight_qmax, self.int_weights, self.proj_set
        
    def quantize_activation_absmax(self, t, act_scales):
        if self.training:
            assert torch.any(torch.isinf(t)) == False, "Input activation contains Inf values"
            assert torch.any(torch.isnan(t)) == False, "Input activation contains NaN values"
            t = torch.div(t, act_scales)
            q_t = t + (torch.round(t) - t).detach()
            q_t = torch.clamp(q_t, min=-self.act_qmax-1, max=self.act_qmax)
            return q_t
        else:
            with torch.no_grad():
                q_t = torch.div(t, act_scales)
                q_t.round_()
                q_t.clamp_(min=-self.act_qmax-1, max=self.act_qmax)
                return q_t
    
    def initialize_act_scales(self, x):
        x = x.permute(0, 2, 1)
        activation_abs_max = x.reshape(-1, x.shape[-1]).abs().max(dim=0, keepdim=False)[0].clamp(min=1e-5)
        act_scales = activation_abs_max
        act_scales = act_scales if self.act_per_channel else act_scales.data.max()
        act_scales = act_scales.view(1, -1, 1)
        act_scales = torch.div(act_scales, self.act_qmax)
        self.act_scales = nn.Parameter(act_scales, requires_grad=False)
        
    def update_act_scales(self, x):
        with torch.no_grad():
            activation_abs_max = x.abs().max(dim=-1)[0]
            activation_abs_max.clamp_(min=1e-5)
            if self.act_per_channel:
                act_scales = activation_abs_max.max(dim=0, keepdim=True)[0].view(1, -1, 1)
            else:
                act_scales = activation_abs_max.max().view(1, 1, 1)
            act_scales = torch.div(act_scales, self.act_qmax)
            
            if not self.training:
                if 'act_scales' in self._parameters:
                    del self._parameters['act_scales']
                self.act_scales = act_scales
            else:
                self.act_scales = nn.Parameter(act_scales, requires_grad=False)
    
    def forward(self, x):
        if not self.ptq:
            return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        ## Activation quantization
        if self.static_quant:
            if self.act_scales is None:
                self.initialize_act_scales(x)
        else:
            self.update_act_scales(x)
        q_x = self.quantize_activation_absmax(x, self.act_scales)
        
        ## Forward pass
        if self.static_quant and self._weight_dequant_scaled_cache is not None:
            return F.conv1d(q_x, self._weight_dequant_scaled_cache, self.bias, self.stride, self.padding, self.dilation, self.groups)

        act_scales_flat = self.act_scales.view(-1)
        if act_scales_flat.numel() == 1:
            weight_dequant_scaled = self.weight_dequant * act_scales_flat.view(1, 1, 1)
        else:
            in_per_group = self.in_channels // self.groups
            act_scales_groups = act_scales_flat.view(self.groups, in_per_group)
            act_scales_for_weight = act_scales_groups[self._out_channel_group_idx.to(device=x.device)].unsqueeze(2)
            weight_dequant_scaled = self.weight_dequant * act_scales_for_weight

        if self.static_quant:
            self._weight_dequant_scaled_cache = weight_dequant_scaled

        y = F.conv1d(q_x, weight_dequant_scaled, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return y

    def __repr__(self):
        return (
            f"QuantConv1d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, bias={self.bias is not None}, "
            f"weight_bits={self.weight_bits}, act_bits={self.act_bits}, "
            f"act_per_channel={self.act_per_channel}, "
            f"power={self.power}, additive={self.additive})"
        )
