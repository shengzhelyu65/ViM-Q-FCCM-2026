import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from model.quant_utils import build_power_value, round_to_power_of_2
except ImportError:
    from quant_utils import build_power_value, round_to_power_of_2

class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scales, weight_qmax, proj_set, int_weights, power, weight_bits, return_int_weights=False):
        if power:
            w_norm = torch.div(weight, scales).clamp(-1, 1)
            sign = w_norm.sign()
            
            # Find closest positive value
            dist = (proj_set.unsqueeze(1).unsqueeze(2) - w_norm.abs().unsqueeze(0)).abs()
            idx = dist.argmin(dim=0)
            
            q_w = proj_set[idx] * sign
            
            if return_int_weights:
                q_w_int = int_weights[idx].clone()
                negative_mask = (sign < 0)
                sign_bit_pos = weight_bits - 1
                q_w_int[negative_mask] |= (1 << sign_bit_pos)
                return q_w, q_w_int
            else:
                return q_w
        else:
            w = torch.div(weight, scales)
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

class QuantLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_bits=5,
        act_bits=8,
        per_block=False,
        block_size=32,
        act_per_token=True,
        power=True,
        additive=True,
        ptq=False,
        static_quant=False,
        layer_name=""
    ):
        super().__init__(in_features, out_features, bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.per_block = per_block
        self.block_size = block_size
        self.act_per_token = act_per_token
        self.power = power
        self.additive = additive
        self.ptq = ptq
        self.static_quant = static_quant
        self.layer_name = layer_name
        self.seq_len = 197
        self.weight_qmax = 2 ** (self.weight_bits - 1) - 1
        self.act_qmax = 2 ** (self.act_bits - 1) - 1
        
        self.calibrated = False
        self.register_buffer("smooth_scales", None)

        if self.per_block:
            if self.in_features < self.block_size:
                self.block_size = self.in_features
            
            if self.in_features % self.block_size != 0:
                original_block_size = self.block_size
                for d in range(self.block_size, 0, -1):
                    if self.in_features % d == 0:
                        self.block_size = d
                        break
                print(f"Warning: layer {self.layer_name} in_features {self.in_features} not divisible by block_size {original_block_size}. Adjusting block_size to {self.block_size}")
            
            assert self.in_features % self.block_size == 0, \
                f"Input features {self.in_features} must be divisible by block size {self.block_size}"
    
    @torch.no_grad()
    def get_weight_scales(self):
        
        if self.per_block:
            w_reshaped = self.weight.reshape(self.out_features, -1, self.block_size)
            weight_abs_max = w_reshaped.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-5)
            weight_scales_blocked = weight_abs_max
            
            weight_scales_blocked = weight_scales_blocked if self.power else torch.div(weight_scales_blocked, self.weight_qmax)
            weight_scales = weight_scales_blocked.squeeze(2)
            
        else:
            weight_abs_max = self.weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
            weight_scales = weight_abs_max
            weight_scales = weight_scales if self.power else torch.div(weight_scales, self.weight_qmax)
            if weight_scales.dim() == 0:
                weight_scales = weight_scales.view(1, 1)
            elif weight_scales.dim() == 1:
                weight_scales = weight_scales.unsqueeze(1)
        
        return weight_scales
    
    @torch.no_grad()
    def from_float(self, float_module):
        self.weight.data.copy_(float_module.weight.data)
        if float_module.bias is not None:
            self.bias.data.copy_(float_module.bias.data)
        else:
            self.bias = None
        
        self.register_buffer("smooth_scales", float_module.smooth_scales if hasattr(float_module, 'smooth_scales') else None)

        self.finalize_calibration()
    
    @torch.no_grad()
    def finalize_calibration(self, calibration_x=None):
        # 1. Compute Scales
        scales = self.get_weight_scales()
        self.register_buffer("weight_scales", scales)
        
        if self.per_block:
            scales_expanded = scales.unsqueeze(2).repeat_interleave(self.block_size, dim=2).reshape(self.out_features, self.in_features)
        else:
            scales_expanded = scales
            
        self.act_scales = None
        
        # Get power-of-two bases and bit allocation
        proj_set, proj_scale, int_weights, base_a, base_b, base_c, bits_allocation = build_power_value(
            B=self.weight_bits - 1, additive=self.additive
        )
        self.register_buffer("proj_set", proj_set.to(self.weight.device))
        self.register_buffer("proj_scale", proj_scale.to(self.weight.device))
        self.register_buffer("int_weights", int_weights.to(self.weight.device))
        
        # 2. Pre-compute quantized weights for PTQ mode
        q_w, q_w_int = QuantizeSTE.apply(
            self.weight, scales_expanded, self.weight_qmax,
            self.proj_set, self.int_weights,
            self.power, self.weight_bits, True
        )
        # Store quantized weights (without scaling) for integer-like forward pass
        self.register_buffer("weight_quant", q_w.to(self.weight.device))
        self.register_buffer("weight_quant_pot", q_w_int.to(self.weight.device))

        if self.per_block:
            scales_expanded_for_forward = self.weight_scales.repeat_interleave(self.block_size, dim=1)
            self.register_buffer("weight_dequant", (self.weight_quant * scales_expanded_for_forward).to(self.weight.device))
        else:
            self.register_buffer("weight_dequant", (self.weight_quant * self.weight_scales).to(self.weight.device))
        
        self.calibrated = True
        
        if self.ptq:
            del self.weight, self.weight_qmax, self.int_weights, self.proj_set
        
    def quantize_activation_absmax(self, t, act_scales):
        if self.training:
            assert torch.any(torch.isinf(t)) == False, "Input activation contains Inf values"
            assert torch.any(torch.isnan(t)) == False, "Input activation contains NaN values"
            t = torch.div(t, act_scales)
            # Use STE (Straight-Through Estimator) for gradients during training
            q_t = t + (torch.round(t) - t).detach()
            q_t = torch.clamp(q_t, min=-self.act_qmax-1, max=self.act_qmax)
            return q_t
        else:
            with torch.no_grad():
                q_t = torch.div(t, act_scales)
                q_t.round_()
                q_t.clamp_(min=-self.act_qmax-1, max=self.act_qmax)
                return q_t
    
    def smooth_x(self, x):
        assert torch.any(torch.isinf(x)) == False, "Input contains Inf values"
        assert torch.any(torch.isnan(x)) == False, "Input contains NaN values"
        x = torch.mul(x, self.smooth_scales.reshape(1, 1, -1))
        return x
    
    def initialize_act_scales(self, x, round_to_power=True):
        x = x.permute(0, 2, 1)
        activation_abs_max = x.reshape(-1, x.shape[-1]).abs().max(dim=0, keepdim=False)[0].clamp(min=1e-5)
        act_scales = round_to_power_of_2(activation_abs_max) if round_to_power else activation_abs_max
        act_scales = act_scales if self.act_per_token else act_scales.data.max()
        act_scales = torch.div(act_scales, self.act_qmax)
        act_scales = act_scales.view(1, -1, 1)
        self.act_scales = nn.Parameter(act_scales, requires_grad=False)
        
    def update_act_scales(self, x):
        with torch.no_grad():
            activation_abs_max = x.abs().max(dim=-1, keepdim=True)[0]
            activation_abs_max.clamp_(min=1e-5)
            if not self.act_per_token:
                activation_abs_max = activation_abs_max.max().view(1, 1, 1)
            act_scales = activation_abs_max / self.act_qmax
            if not self.training:
                if 'act_scales' in self._parameters:
                    del self._parameters['act_scales']
                self.act_scales = act_scales
            else:
                self.act_scales = nn.Parameter(act_scales, requires_grad=False)

    def forward(self, x):
        x_dim = x.dim()
        x = x.unsqueeze(1) if x_dim == 2 else x
        s_x = self.smooth_x(x) if self.smooth_scales is not None else x

        if not self.ptq:
            s_x = s_x.squeeze(1) if x_dim == 2 else s_x
            return F.linear(s_x, self.weight, self.bias)
        
        ## Activation quantization
        if self.static_quant:
            if self.act_scales is None:
                self.initialize_act_scales(s_x)
        else:
            self.update_act_scales(s_x)
        q_x = self.quantize_activation_absmax(s_x, self.act_scales)
        
        ## Forward pass
        y = F.linear(q_x, self.weight_dequant, None)
        y = y * self.act_scales

        if self.bias is not None:
            y = y + self.bias

        y = y.squeeze(1) if x_dim == 2 else y
        return y

    def __repr__(self):
        return (
            f"QuantLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"weight_bits={self.weight_bits}, act_bits={self.act_bits}, "
            f"per_block={self.per_block}, block_size={self.block_size}, "
            f"act_per_token={self.act_per_token}, "
            f"power={self.power}, additive={self.additive})"
        )
