import torch
import os

import argparse
import torch.nn as nn
import numpy as np

from timm.models import create_model
from datasets import build_dataset
import functools
import torch.backends.cudnn as cudnn

import utils as utils
import models_mamba

def get_act_scales(model, dataloader):
    model.eval()
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        # Computes the maximum value along dimension 0 (across all elements for each feature), returning a tuple (values, indices).
        comming_max = torch.max(tensor, dim=0)[0].float().cpu() # [hidden_dim]
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

        if 'layers.0.' in name:
            print(f"Processed {name}: {act_scales[name].shape}, max={act_scales[name].max().item()}")

    def stat_input_hook(m, x, y, name): # m is the module, x is the input, y is the output and name is the name of the module
        if isinstance(x, tuple):
            # print(f"Processed tuple: {name}")
            x = x[0]
        if 'conv' in name:
            x = x.transpose(1, 2)
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))
        
        if isinstance(m, nn.Conv1d):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for _, (input, _) in enumerate(dataloader):
        input = input.to('cuda')
        print(f"Batch size: {input.shape[0]}")
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                output = model.module(input)
            else:
                output = model(input)
        break

    for h in hooks:
        h.remove()

    return act_scales

def get_args_parser():
    parser = argparse.ArgumentParser('ViM smooth scales script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--scales-output-path', type=str, default='./',
                        help='where to save the act scales')
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local-rank', default=0, type=int)
    
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    return parser

@torch.no_grad()
def main(args):
    utils.init_distributed_mode(args)
    print(args)
    
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
       
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )
    print(model)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=True)
        print(msg)
        
    output_root = os.path.abspath(args.scales_output_path)
    if os.path.basename(os.path.normpath(output_root)) == "act_scales":
        scale_dir = output_root
    else:
        scale_dir = os.path.join(output_root, "act_scales")

    # Save scale files with model name in filename
    os.makedirs(scale_dir, exist_ok=True)
    scale_base = args.model
    save_path = os.path.join(scale_dir, f'{scale_base}_smoothing.pt')
    act_scales = get_act_scales(model_without_ddp, data_loader_train)
    torch.save(act_scales, save_path)
    print(f"Saved activation scales to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate activation scales', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
