import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import selective_scan_cuda
from torch.cuda.amp import custom_bwd, custom_fwd
from model.ops.scan_ref import manual_scan

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
    return out, rearrange(delta, "b d l -> l b d"), lambda_elements, bu_elements, xs

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, conv1d_out, delta, A, B, C, D, z, delta_bias=None, delta_softplus=True):
        
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        
        ctx.save_for_backward(conv1d_out, delta, A, B, C, D, z, delta_bias, scan_intermediates, out)

        return out_z
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out_z):
        conv1d_out, delta, A, B, C, D, z, delta_bias, scan_intermediates, out = ctx.saved_tensors
        
        grad_out_z = grad_out_z.contiguous()
        dz = torch.empty_like(grad_out_z).contiguous()
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, grad_out_z, scan_intermediates, out, dz,
            True, # delta_softplus
            True  # option to recompute out_z
        )
        
        return dconv1d_out, ddelta, dA, dB, dC, dD, dz, ddelta_bias, None, None

class MambaModule(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        bias=False,
        device=None,
        dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.act = nn.SiLU()
        
        # A
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True
        
        # in_proj
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        
        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # x_proj and dt_proj
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        dt_scale = 1.0
        dt_max = 0.1
        dt_min = 0.001
        dt_init_floor = 1e-4
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(self, hidden_states, manual_scan=False):
        dtype = hidden_states.dtype
        
        ## A and D
        A = -torch.exp(self.A_log.float()).to(dtype)
        A_b = -torch.exp(self.A_b_log.float()).to(dtype)
        
        D = self.D.float().to(dtype).contiguous()
        D_b = self.D_b.float().to(dtype).contiguous()
        
        ## Calculate x and z
        xz = self.in_proj(hidden_states) # [B L D]
        xz_b = xz.flip([1])
        
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        x, z = xz.chunk(2, dim=2)
        if xz_b.stride(-1) != 1:
            xz_b = xz_b.contiguous()
        x_b, z_b = xz_b.chunk(2, dim=2)
        
        ## Causal Conv1d
        x = x.permute(0, 2, 1) # [B D L]
        conv_out = self.conv1d(x)[:, :, :x.shape[-1]]
        conv_out = self.act(conv_out)
        conv_out = conv_out.permute(0, 2, 1) # [B L D]
        
        x_b = x_b.permute(0, 2, 1) # [B D L]
        conv_out_b = self.conv1d_b(x_b)[:, :, :x_b.shape[-1]]
        conv_out_b = self.act(conv_out_b)
        conv_out_b = conv_out_b.permute(0, 2, 1)
        
        ## X Projection and Dt Projection
        x_bld = self.x_proj(conv_out) # [B L D]
        delta = self.dt_proj(x_bld[:, :, :self.dt_rank]) # [B L D]
        
        x_bld_b = self.x_proj_b(conv_out_b) # [B L D]
        delta_b = self.dt_proj_b(x_bld_b[:, :, :self.dt_rank]) # [B L D]
        
        ## Get B and C
        B = x_bld[:, :, self.dt_rank:self.dt_rank + self.d_state]
        B = rearrange(B, "b l dstate -> b 1 dstate l").contiguous()
        C = x_bld[:, :, -self.d_state:]
        C = rearrange(C, "b l dstate -> b 1 dstate l").contiguous()
        
        B_b = x_bld_b[:, :, self.dt_rank:self.dt_rank + self.d_state]
        B_b = rearrange(B_b, "b l dstate -> b 1 dstate l").contiguous()
        C_b = x_bld_b[:, :, -self.d_state:]
        C_b = rearrange(C_b, "b l dstate -> b 1 dstate l").contiguous()
        
        ## Selective Scan
        conv_out_var = conv_out.permute(0, 2, 1).contiguous() # [B D L]
        delta_var = delta.permute(0, 2, 1).contiguous() # [B D L]
        z_var = z.permute(0, 2, 1).contiguous() # [B D L]
        # B: [B 1 D L], C: [B 1 D L]
        if not manual_scan:
            out_z = SelectiveScanFn.apply( # [B D L]
                conv_out_var, delta_var, A, B, C, D, z_var, None, True
            )
        else:
            out_z, delta, lambda_elements, bu_elements, xs = selective_scan_parallel(
                conv_out_var, delta_var, A, B, C, D, z=z_var
            )
            self.last_delta = delta
            self.last_B = rearrange(B, "b 1 dstate l -> l b dstate")
            self.last_C = rearrange(C, "b 1 dstate l -> l b dstate")
            self.last_lambda_elements = lambda_elements
            self.last_bu_elements = bu_elements
            self.last_xs = xs
            self.last_out_z = rearrange(out_z, "b d l -> b l d")
        
        conv_out_b_var = conv_out_b.permute(0, 2, 1).contiguous() # [B D L]
        delta_b_var = delta_b.permute(0, 2, 1).contiguous() # [B D L]
        z_b_var = z_b.permute(0, 2, 1).contiguous() # [B D L]
        # B: [B 1 D L], C: [B 1 D L]
        if not manual_scan:
            out_z_b = SelectiveScanFn.apply( # [B D L]
                conv_out_b_var, delta_b_var, A_b, B_b, C_b, D_b, z_b_var, None, True
            )
        else:
            out_z_b, delta_b, lambda_elements_b, bu_elements_b, xs_b = selective_scan_parallel(
                conv_out_b_var, delta_b_var, A_b, B_b, C_b, D_b, z=z_b_var
            )
            self.last_delta_b = delta_b
            self.last_B_b = rearrange(B_b, "b 1 dstate l -> l b dstate")
            self.last_C_b = rearrange(C_b, "b 1 dstate l -> l b dstate")
            self.last_lambda_elements_b = lambda_elements_b
            self.last_bu_elements_b = bu_elements_b
            self.last_xs_b = xs_b
            self.last_out_z_b = rearrange(out_z_b, "b d l -> b l d")
        
        ## Output Projection
        out_z = rearrange(out_z + out_z_b.flip([-1]), "b d l -> b l d") / 2
        out = self.out_proj(out_z) # [B L D]
        
        return out