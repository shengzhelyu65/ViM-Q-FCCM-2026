import torch
import torch.nn as nn
from model.quant_linear import QuantLinear
from model.quant_conv1d import QuantConv1d

class ActivationHook:
    def __init__(self):
        self.activations = {}
        self.handles = []
        
    def hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = input[0].detach()
        return hook
    
    def register_hooks(self, model):
        self.remove_hooks()
        
        for name, module in model.named_modules():
            if isinstance(module, (QuantLinear, QuantConv1d)):
                handle = module.register_forward_hook(self.hook_fn(name))
                self.handles.append(handle)
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

def initialize_act_scales(model, data_loader, device, sample_size=4):
    sample_inputs = None
    for i, (inputs, _) in enumerate(data_loader):
        if i == 0:
            sample_inputs = inputs
        else:
            sample_inputs = torch.cat((sample_inputs, inputs), dim=0)
        if sample_inputs.shape[0] >= sample_size:
            break
    sample_inputs = sample_inputs[:sample_size].to(device)
    print(f"Sample inputs shape: {sample_inputs.shape}")
    
    training_mode = model.training
    model.eval()
    
    act_hook = ActivationHook()
    act_hook.register_hooks(model)
    
    with torch.no_grad():
        if not isinstance(sample_inputs, list):
            sample_inputs = [sample_inputs]
            
        for sample_input in sample_inputs:
            _ = model(sample_input)
    
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantConv1d)):
            if name in act_hook.activations:
                act = act_hook.activations[name]
                if isinstance(module, QuantLinear):
                    act = act.unsqueeze(1) if act.dim() == 2 else act
                    act = module.smooth_x(act) if module.smooth_scales is not None else act
                module.initialize_act_scales(act)
            else:
                print(f"Warning: No activation captured for layer {name}")
    
    act_hook.remove_hooks()
    
    if training_mode:
        model.train()
    
    return model

def quantize_vim_model(
    model,
    weight_bits=4, act_bits=8,
    linear_act_per_token=True,
    conv_act_per_channel=False,
    power=True, additive=True,
    per_block=True, block_size=32,
    quantize_head=True,
    ptq=True,
    static_quant=False
):
    from models_mamba import Block
    
    print(f"Quantizing model {model.__class__.__name__} with static_quant={static_quant}, ptq={ptq}")
    
    layer_suffixes = [
        "in_proj",
        "conv1d",
        "x_proj",
        "dt_proj",
        "conv1d_b",
        "x_proj_b",
        "dt_proj_b",
        "out_proj"
    ]
        
    for name, module in model.named_modules():
        if isinstance(module, Block):
            for suffix in layer_suffixes:
                layer = getattr(module.mixer, suffix)
                if isinstance(layer, nn.Linear):
                    layer_name = f"{name}.mixer.{suffix}"
                    quantized_layer = QuantLinear(
                        layer.in_features,
                        layer.out_features,
                        bias=layer.bias is not None,
                        weight_bits=weight_bits,
                        act_bits=act_bits,
                        act_per_token=linear_act_per_token,
                        power=power,
                        additive=additive,
                        per_block=per_block,
                        block_size=block_size,
                        ptq=ptq,
                        static_quant=static_quant,
                        layer_name=layer_name
                    )
                    quantized_layer.from_float(layer)
                    quantized_layer = quantized_layer.to(layer.weight.device)
                    delattr(module.mixer, suffix)
                    setattr(module.mixer, suffix, quantized_layer)
                
                elif isinstance(layer, nn.Conv1d):
                    layer_name = f"{name}.mixer.{suffix}"
                    quantized_layer = QuantConv1d(
                        layer.in_channels,
                        layer.out_channels,
                        kernel_size=layer.kernel_size[0],
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                        bias=layer.bias is not None,
                        weight_bits=5,
                        act_bits=8,
                        act_per_channel=conv_act_per_channel,
                        power=power,
                        additive=additive,
                        ptq=ptq,
                        static_quant=static_quant,
                        layer_name=layer_name
                    )
                    quantized_layer.from_float(layer)
                    quantized_layer = quantized_layer.to(layer.weight.device)
                    delattr(module.mixer, suffix)
                    setattr(module.mixer, suffix, quantized_layer)
                    
                else:
                    raise ValueError(f"Unsupported layer type: {type(layer)}")
        
        elif isinstance(module, nn.Linear) and 'head' in name and quantize_head:
            quantized_layer = QuantLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                weight_bits=weight_bits,
                act_bits=act_bits,
                act_per_token=linear_act_per_token,
                power=power,
                additive=additive,
                per_block=per_block,
                block_size=block_size,
                ptq=ptq,
                static_quant=static_quant,
                layer_name=name
            )
            quantized_layer.from_float(module)
            quantized_layer = quantized_layer.to(module.weight.device)
            delattr(model, name)
            setattr(model, name, quantized_layer)
            
        else:
            continue
    
    ## Print the trainable parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable parameter: {name}, Shape: {param.shape}")
    return model
