import math
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import to_2tuple, DropPath

from model.mamba_simple_module import MambaModule as Mamba

try:
    from model.ops.triton.layernorm import RMSNorm, rms_norm_fn
except ImportError:
    RMSNorm, rms_norm_fn = None, None


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            import warnings
            warnings.warn(f"Input image size ({H}*{W}) differs from model default ({self.img_size[0]}*{self.img_size[1]}). "
                         f"This may affect model performance but allows profiling at different resolutions.")
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, residual_in_fp32=True, drop_path=0., manual_scan=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.manual_scan = manual_scan

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None
    ):
        if residual is None:
            hidden_states, residual = rms_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        else:
            hidden_states, residual = rms_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, manual_scan=self.manual_scan)
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    d_state=16,
    norm_epsilon=1e-5,
    drop_path=0.,
    residual_in_fp32=True,
    layer_idx=None,
    device=None,
    dtype=None,
    manual_scan=False
):
    ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state=d_state, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        residual_in_fp32=residual_in_fp32,
        manual_scan=manual_scan,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class VisionMambaQuant(nn.Module):
    def __init__(self, 
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        d_state=16,
        channels=3,
        num_classes=1000,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        manual_scan=False,
        max_layers=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()
        
        self.residual_in_fp32 = residual_in_fp32
        self.num_tokens = 1
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.max_layers = max_layers  # Limit number of layers to process (None = all layers)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
            
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    residual_in_fp32=residual_in_fp32,
                    layer_idx=i,
                    drop_path=inter_dpr[i],
                    manual_scan=manual_scan,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        self.norm_f = RMSNorm(embed_dim, eps=norm_epsilon, **factory_kwargs)
        
        self.head = nn.Linear(self.num_features, num_classes)
        
        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # mamba init
        initializer_cfg = None
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
        
    def _interpolate_pos_embedding(self, pos_embed, new_size):
        """
        Interpolate positional embeddings for different input sizes.
        This allows the model to work with different resolutions during profiling.
        """
        # Get original and new sizes
        orig_size = pos_embed.shape[1]
        
        if orig_size == new_size:
            return pos_embed
        
        # Simple interpolation: repeat or truncate
        if new_size > orig_size:
            # If new size is larger, repeat the embeddings
            repeat_factor = (new_size + orig_size - 1) // orig_size  # Ceiling division
            pos_embed_expanded = pos_embed.repeat(1, repeat_factor, 1)
            return pos_embed_expanded[:, :new_size, :]
        else:
            # If new size is smaller, truncate
            return pos_embed[:, :new_size, :]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, M, _ = x.shape

        # Add cls_token
        cls_token = self.cls_token.expand(B, -1, -1)
        token_position = M // 2
        x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
        M = x.shape[1]

        # Add position embedding with interpolation for different input sizes
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolate positional embeddings for different input sizes
            pos_embed_resized = self._interpolate_pos_embedding(self.pos_embed, x.shape[1])
            x = x + pos_embed_resized
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Mamba Implementation
        residual = None
        hidden_states = x
        layers_to_process = self.layers[:self.max_layers] if self.max_layers is not None else self.layers
        for layer in layers_to_process:
            hidden_states, residual = layer(
                hidden_states, residual
            )

        # Final RMSNorm
        hidden_states = rms_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )

        # Return the cls token
        return hidden_states[:, token_position, :]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def vim_tiny_pretrain(**kwargs):
    model = VisionMambaQuant(patch_size=16, embed_dim=192, depth=24, residual_in_fp32=False, **kwargs)
    return model

@register_model
def vim_small_pretrain(**kwargs):
    model = VisionMambaQuant(patch_size=16, embed_dim=384, depth=24, residual_in_fp32=False, **kwargs)
    return model

@register_model
def vim_base_pretrain(**kwargs):
    model = VisionMambaQuant(patch_size=16, embed_dim=768, depth=24, residual_in_fp32=False, **kwargs)
    return model

@register_model
def vim_tiny_finetune(pretrained=False, **kwargs):
    model = VisionMambaQuant(patch_size=16, stride=8, embed_dim=192, depth=24, residual_in_fp32=False, **kwargs)
    return model

@register_model
def vim_small_finetune(pretrained=False, **kwargs):
    model = VisionMambaQuant(patch_size=16, stride=8, embed_dim=384, depth=24, residual_in_fp32=False, **kwargs)
    return model