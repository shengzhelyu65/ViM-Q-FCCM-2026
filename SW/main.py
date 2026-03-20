import argparse
import datetime
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from contextlib import suppress

from timm.data import Mixup
from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_dataset
from engine import evaluate, train_one_epoch
from samplers import RASampler
from losses import DistillationLoss
import utils as utils

from b_smooth_model import smooth_vim_model
from c_quantize_model import quantize_vim_model, initialize_act_scales

import models_mamba
from model.quant_linear import QuantLinear
from model.quant_conv1d import QuantConv1d

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

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

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='output',
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

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=False)

    # if continue with inf
    parser.add_argument('--if_continue_inf', action='store_true')
    parser.add_argument('--no_continue_inf', action='store_false', dest='if_continue_inf')
    parser.set_defaults(if_continue_inf=False)

    # if use nan to num
    parser.add_argument('--if_nan2num', action='store_true')
    parser.add_argument('--no_nan2num', action='store_false', dest='if_nan2num')
    parser.set_defaults(if_nan2num=False)

    # if use random token position
    parser.add_argument('--if_random_cls_token_position', action='store_true')
    parser.add_argument('--no_random_cls_token_position', action='store_false', dest='if_random_cls_token_position')
    parser.set_defaults(if_random_cls_token_position=False)    

    # if use random token rank
    parser.add_argument('--if_random_token_rank', action='store_true')
    parser.add_argument('--no_random_token_rank', action='store_false', dest='if_random_token_rank')
    parser.set_defaults(if_random_token_rank=False)

    parser.add_argument('--local-rank', default=0, type=int)
    
    # Debug
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    
    # Quantization options
    parser.add_argument("--act_scales", default='', help='path for act_scale checkpoint')
    parser.add_argument('--manual_scan', action='store_true', default=False, help='Use manual scan in MambaModule forward pass')
    parser.add_argument('--static_quant', action='store_true', help='Use static activation quantization under PTQ')
    parser.add_argument('--ptq', action='store_true', help='Perform post-training quantization')
    parser.add_argument('--smooth', action='store_true', help='Use smooth quantization')
    parser.add_argument('--linear-act-per-token', type=str2bool, default=True, help='Use per-token quantization for linear activations')
    parser.add_argument('--conv-act-per-channel', type=str2bool, default=False, help='Use per-channel quantization for conv activations')
    parser.add_argument('--weight_bits', type=int, default=4, help='Number of bits for weight quantization')
    parser.add_argument('--additive', type=str2bool, default=True, help='Use additive quantization')
    parser.add_argument('--power', type=str2bool, default=True, help='Use power quantization')
    parser.add_argument('--quantize_head', type=str2bool, default=True, help='Quantize head')
    parser.add_argument('--per-block', action='store_true', help='Use per-block quantization')
    parser.add_argument('--block-size', type=int, default=32, help='Block size for per-block quantization')
    return parser

def main(args):
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    
    ## Data Loader
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    ## Debugging
    if args.debug:
        if args.eval:
            debug_size_val = min(len(dataset_val), 10)          # Use only 10 samples or fewer for validation
            dataset_val = torch.utils.data.Subset(dataset_val, range(debug_size_val))
            
        else:
            debug_size_train = min(len(dataset_train), 500)     # Use only 500 samples or fewer for training
            debug_size_val = min(len(dataset_val), 200)         # Use only 200 samples or fewer for validation
            dataset_train = torch.utils.data.Subset(dataset_train, range(debug_size_train))
            dataset_val = torch.utils.data.Subset(dataset_val, range(debug_size_val))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
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
    
    ## Data Augmentation of Mixup and Cutmix (used)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        manual_scan=args.manual_scan
    )
    model.to(device)
    
    ### Load checkpoint BEFORE creating DDP
    if args.resume:
        if utils.is_main_process():
            print(f"Loading checkpoint from {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'vim_b' in args.resume:
            state_dict = checkpoint['model_ema']
        else:
            state_dict = checkpoint['model']
        
        model.load_state_dict(state_dict, strict=True)
        
        ## Apply smoothing and quantization BEFORE DDP
        if args.smooth:
            if utils.is_main_process():
                print("Applying smoothing and quantization...")
            
            # Load act_scales only on main process, then broadcast
            if utils.is_main_process():
                # Use model name to select correct scale file
                scale_file = f"{args.model}_smoothing.pt"
                if '.pt' not in args.act_scales:
                    act_scales_path = f"{args.act_scales}/{scale_file}"
                else:
                    act_scales_path = args.act_scales
                act_scales = torch.load(act_scales_path, map_location='cpu')
            else:
                act_scales = None
            
            # Broadcast act_scales to all processes in distributed mode
            if args.distributed:
                import torch.distributed as dist
                if utils.is_main_process():
                    # Serialize act_scales
                    import pickle
                    act_scales_bytes = pickle.dumps(act_scales)
                    act_scales_tensor = torch.ByteTensor(list(act_scales_bytes)).to(device)
                    size_tensor = torch.tensor([len(act_scales_bytes)], dtype=torch.long).to(device)
                else:
                    size_tensor = torch.tensor([0], dtype=torch.long).to(device)
                
                # Broadcast size first
                dist.broadcast(size_tensor, src=0)
                
                if not utils.is_main_process():
                    act_scales_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8).to(device)
                
                # Broadcast data
                dist.broadcast(act_scales_tensor, src=0)
                
                if not utils.is_main_process():
                    import pickle
                    act_scales_bytes = bytes(act_scales_tensor.cpu().numpy())
                    act_scales = pickle.loads(act_scales_bytes)
            
            # Apply smoothing
            smooth_vim_model(model, act_scales, dtype=torch.float32, alpha=0.5)
            
            # Apply quantization
            ptq_enabled = args.ptq or args.static_quant
            quantize_vim_model(
                model,
                weight_bits=args.weight_bits,
                act_bits=8,
                linear_act_per_token=args.linear_act_per_token,
                conv_act_per_channel=args.conv_act_per_channel,
                power=args.power,
                additive=args.additive,
                per_block=args.per_block,
                block_size=args.block_size,
                quantize_head=args.quantize_head,
                ptq=ptq_enabled,
                static_quant=args.static_quant
            )
            print("Quantization Parameters:")
            print(f"  Linear activation per token: {args.linear_act_per_token}")
            print(f"  Convolution activation per channel: {args.conv_act_per_channel}")
            print(f"  Power: {args.power}")
            print(f"  Additive: {args.additive}")
            print(f"  Quantize head: {args.quantize_head}")

            # Synchronize all processes before proceeding
            if args.distributed:
                dist.barrier()
                if utils.is_main_process():
                    print("All processes synchronized after quantization")
            
            # Initialize act_scales for static activation quantization
            if ptq_enabled and args.static_quant:
                initialize_act_scales(model, data_loader_train, device)
                
            # Synchronize again after act_scales initialization
            if args.distributed:
                dist.barrier()
    
    # Create DDP wrapper AFTER all model modifications
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu], 
            find_unused_parameters=True,
            broadcast_buffers=True  # Ensure buffers are synchronized
        )
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if utils.is_main_process():
        print('Number of parameters:', n_parameters)
    
    ## Scale learning rate (used)
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    
    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = SoftTargetCrossEntropy()
    criterion = DistillationLoss(
        criterion, None, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    
    output_dir = Path(args.output_dir)
    
    ### Evaluation
    if args.eval:
        # Ensure model is in eval mode across all processes
        model.eval()
        
        # Synchronize before evaluation in distributed mode
        if args.distributed:
            torch.distributed.barrier()
            if utils.is_main_process():
                print("Starting distributed evaluation...")
        
        ## Evaluation with proper distributed handling
        test_stats = evaluate(data_loader_val, model, device)
        
        if utils.is_main_process():
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vim quantization script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
