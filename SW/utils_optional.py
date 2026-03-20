import json
import shutil
import time
import threading
import platform
import subprocess
import statistics
from pathlib import Path
from typing import Dict

import numpy as np
import psutil
import torch
import torch.distributed as dist


def _is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def _get_rank():
    if not _is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return _get_rank() == 0

def interpolate_pos_embed(model, state_dict):
    pos_embed_checkpoint = state_dict['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # import ipdb; ipdb.set_trace()
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed
        
def print_parameter_table(model):
    print("Parameter table:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")
    print("Total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
def start_logger(args, output_dir):
    if args.output_dir and is_main_process():
        backup_dir = output_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy("models_mamba.py", str(backup_dir / "models_mamba.py"))
        shutil.copy("b_smooth_model.py", str(backup_dir / "b_smooth_model.py"))
        shutil.copy("c_quantize_model.py", str(backup_dir / "c_quantize_model.py"))
        shutil.copy("model/mamba_simple_module.py", str(backup_dir / "mamba_simple_module.py"))
        shutil.copy("model/quant_linear.py", str(backup_dir / "quant_linear.py"))
        shutil.copy("model/quant_conv1d.py", str(backup_dir / "quant_conv1d.py"))
            
        with open(output_dir / "log.txt", "w") as f:
            f.write("Training log\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Warmup epochs: {args.warmup_epochs}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"Warmup learning rate: {args.warmup_lr}\n")
            f.write(f"Weight decay: {args.weight_decay}\n")
            f.write(f"Dropout rate: {args.drop}\n")
            f.write(f"Drop path rate: {args.drop_path}\n")
            f.write(f"--------------------\n")
            f.write(f"Debug: {args.debug}\n")
            f.write(f"Static quant: {args.static_quant}\n")
            f.write(f"Smooth: {args.smooth}\n")
            f.write(f"--------------------\n")
        
def update_logger(args, output_dir, log_stats):
    if args.output_dir and is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

def make_all_one_act_scale(input_path, output_path):
    """
    Loads an activation scale file, sets all values to 1, and saves to output_path.
    """
    import torch
    act_scales = torch.load(input_path, map_location='cpu')
    all_one_scales = {}
    for k, v in act_scales.items():
        all_one_scales[k] = torch.ones_like(v)
    torch.save(all_one_scales, output_path)
    print(f"Saved all-one activation scales to {output_path}")
            
# =============================================================================
# Timed Model Profiler (using built-in model timing)
# =============================================================================
def profile_timed_model(model, device, output_dir, input_shape=(1, 3, 224, 224), 
                       warmup_runs=20, profile_runs=100, trace_name="timed_model_profile"):
    """
    Profile model using built-in timing mechanisms in timed models.
    
    Args:
        model: The timed model to profile (should be a TimedVisionMamba)
        device: Device to run on  
        output_dir: Directory to save results
        input_shape: Input tensor shape (default: (1, 3, 224, 224))
        warmup_runs: Warmup iterations (default: 20)
        profile_runs: Profile iterations (default: 100)
        trace_name: Output filename prefix
    
    Returns:
        dict: Results with detailed component timing breakdown
    """
    from pathlib import Path
    import json
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Display profiling configuration
    print(f"Input shape: {input_shape}")
    print(f"Profiling: {warmup_runs} warmup + {profile_runs} runs")
    
    # Check if this is a timed model
    if not hasattr(model, 'enable_timing'):
        raise ValueError("Model must be a timed model with enable_timing() method")
    
    # Warmup runs
    print(f"Warming up with {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Enable timing
    model.enable_timing()
    
    # Profile runs
    print(f"Running {profile_runs} profiling iterations...")
    inference_times = []
    component_times = {}
    
    with torch.no_grad():
        for run_idx in range(profile_runs):
            # Reset timing stats
            model.reset_timing_stats()
            
            # Time the overall inference
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                output = model(dummy_input)
                
                torch.cuda.synchronize()
                end_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
                output = model(dummy_input)
                end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Get component times for this run
            timing_stats = model.get_timing_stats()
            
            # Debug: Show sample counts for the first run
            if run_idx == 0:
                print(f"\nDEBUG: Sample counts per component in one run:")
                for component, times in timing_stats.items():
                    print(f"  {component:20}: {len(times):3d} individual measurements")
                print()
            
            for component, times in timing_stats.items():
                if component not in component_times:
                    component_times[component] = []
                # Sum all times for this component in this run
                component_times[component].append(sum(times))
            
            if (run_idx + 1) % 10 == 0:
                print(f"Completed {run_idx + 1}/{profile_runs} runs")
    
    # Disable timing
    model.disable_timing()
    print("Profiling completed.")
    
    # Calculate statistics
    def calculate_statistics(times):
        if not times:
            return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'sum': 0}
        
        times_array = np.array(times)
        return {
            'mean': float(np.mean(times_array)),
            'median': float(np.median(times_array)),
            'std': float(np.std(times_array)),
            'min': float(np.min(times_array)),
            'max': float(np.max(times_array)),
            'sum': float(np.sum(times_array))
        }
    
    # Generate summary
    summary = {
        'overall_inference': calculate_statistics(inference_times),
        'components': {}
    }
    
    total_component_time = 0
    for component, times in component_times.items():
        if times:  # Only include components that were actually measured
            stats = calculate_statistics(times)
            stats['sample_count'] = len(times)  # Add sample count
            summary['components'][component] = stats
            total_component_time += stats['mean']
    
    # Calculate percentage breakdown
    if summary['overall_inference']['mean'] > 0:
        for component in summary['components']:
            component_mean = summary['components'][component]['mean']
            percentage = (component_mean / summary['overall_inference']['mean']) * 100
            summary['components'][component]['percentage'] = percentage
    
    # Add total measured component time vs overall time
    summary['total_measured_component_time'] = total_component_time
    if summary['overall_inference']['mean'] > 0:
        summary['measurement_coverage'] = (total_component_time / summary['overall_inference']['mean']) * 100
    
    # Validation: warn if coverage is unrealistic
    if summary.get('measurement_coverage', 0) > 120:
        print(f"WARNING: Measurement coverage is {summary['measurement_coverage']:.1f}%, indicating possible double-counting")
    
    # Save results
    results_file, summary_file, csv_file = _save_timed_results(summary, output_path, trace_name, input_shape, device, warmup_runs, profile_runs)
    
    # Print summary to console
    _print_timed_summary(summary, results_file, summary_file, csv_file)
    
    return summary


def _save_timed_results(summary, output_path, trace_name, input_shape, device, warmup_runs, profile_runs):
    """Save timed profiling results to file."""
    import json
    
    # Save detailed results as JSON
    results_file = output_path / f"{trace_name}_detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save human-readable summary
    summary_file = output_path / f"{trace_name}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Timed Vision Mamba Model Profiling Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Model input shape: {input_shape}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Warmup runs: {warmup_runs}\n")
        f.write(f"  Profile runs: {profile_runs}\n\n")
        
        f.write("Overall Inference Performance:\n")
        f.write("-" * 30 + "\n")
        inf_stats = summary['overall_inference']
        f.write(f"  Median time:      {inf_stats['median']:.3f} ms\n")
        f.write(f"  Mean time:        {inf_stats['mean']:.3f} ms\n")
        f.write(f"  Std dev:          {inf_stats['std']:.3f} ms\n")
        f.write(f"  Min time:         {inf_stats['min']:.3f} ms\n")
        f.write(f"  Max time:         {inf_stats['max']:.3f} ms\n")
        f.write(f"  Throughput (median): {1000.0 / inf_stats['median']:.1f} FPS\n")
        f.write(f"  Throughput (mean):   {1000.0 / inf_stats['mean']:.1f} FPS\n\n")
        
        f.write("Component Breakdown (sorted by median time):\n")
        f.write("-" * 30 + "\n")
        
        # Sort components by median time (descending)
        sorted_components = sorted(summary['components'].items(), 
                                 key=lambda x: x[1]['median'], reverse=True)
        
        for component, stats in sorted_components:
            median_percentage = (stats['median'] / inf_stats['median']) * 100 if inf_stats['median'] > 0 else 0
            sample_count = stats.get('sample_count', 0)
            f.write(f"  {component.upper()}:\n")
            f.write(f"    Median time: {stats['median']:.3f} ms ({median_percentage:.1f}%)\n")
            f.write(f"    Mean time:   {stats['mean']:.3f} ms ({stats.get('percentage', 0):.1f}%)\n")
            f.write(f"    Std dev:     {stats['std']:.3f} ms\n")
            f.write(f"    Min time:    {stats['min']:.3f} ms\n")
            f.write(f"    Max time:    {stats['max']:.3f} ms\n")
            f.write(f"    Samples:     {sample_count}\n\n")
        
        f.write("Coverage Analysis:\n")
        f.write("-" * 30 + "\n")
        
        # Calculate median coverage
        total_median_component_time = sum(stats['median'] for stats in summary['components'].values())
        median_coverage = (total_median_component_time / inf_stats['median']) * 100 if inf_stats['median'] > 0 else 0
        
        f.write(f"  Total measured component time (mean):   {summary['total_measured_component_time']:.3f} ms\n")
        f.write(f"  Total measured component time (median): {total_median_component_time:.3f} ms\n")
        f.write(f"  Overall inference time (mean):          {inf_stats['mean']:.3f} ms\n")
        f.write(f"  Overall inference time (median):        {inf_stats['median']:.3f} ms\n")
        f.write(f"  Measurement coverage (mean):            {summary.get('measurement_coverage', 0):.1f}%\n")
        f.write(f"  Measurement coverage (median):          {median_coverage:.1f}%\n\n")
        
        f.write("Half-th Epoch (Median) Summary:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Overall median time:    {inf_stats['median']:.3f} ms\n")
        f.write(f"  Median throughput:      {1000.0 / inf_stats['median']:.1f} FPS\n\n")
        f.write("  Component median times:\n")
        for component, stats in sorted_components:
            sample_count = stats.get('sample_count', 0)
            f.write(f"    {component:20}: {stats['median']:8.3f} ms [{sample_count:3d} samples]\n")
    
    # Save CSV file
    csv_file = output_path / f"{trace_name}_component_times.csv"
    with open(csv_file, 'w') as f:
        # Write CSV header
        f.write("component,median_time_ms,mean_time_ms,std_dev_ms,min_time_ms,max_time_ms,median_percentage,mean_percentage,sample_count\n")
        
        # Write component data in a fixed order for consistency across all runs
        # Define a fixed order based on typical component hierarchy
        fixed_order = ['selective_scan', 'mamba_linear', 'mamba_conv', 'block_norm', 
                      'patch_embed', 'norm', 'pos_embed', 'head']
        
        # Create ordered list maintaining the fixed order, but only including components that exist
        ordered_components = []
        for component in fixed_order:
            if component in summary['components']:
                ordered_components.append((component, summary['components'][component]))
        
        # Add any components not in the fixed order (fallback)
        for component, stats in sorted_components:
            if component not in [c[0] for c in ordered_components]:
                ordered_components.append((component, stats))
        
        for component, stats in ordered_components:
            median_percentage = (stats['median'] / inf_stats['median']) * 100 if inf_stats['median'] > 0 else 0
            sample_count = stats.get('sample_count', 0)
            mean_percentage = stats.get('percentage', 0)
            
            f.write(f"{component},{stats['median']:.3f},{stats['mean']:.3f},{stats['std']:.3f},"
                   f"{stats['min']:.3f},{stats['max']:.3f},{median_percentage:.1f},"
                   f"{mean_percentage:.1f},{sample_count}\n")
        
    return results_file, summary_file, csv_file


def _print_timed_summary(summary, results_file, summary_file, csv_file):
    """Print timed profiling summary to console."""
    print("\n" + "=" * 60)
    print("TIMED MODEL PROFILING RESULTS SUMMARY")
    print("=" * 60)
    
    inf_stats = summary['overall_inference']
    print(f"Overall inference time (median): {inf_stats['median']:.3f} ms")
    print(f"Overall inference time (mean):   {inf_stats['mean']:.3f} ± {inf_stats['std']:.3f} ms")
    print(f"Throughput (median): {1000.0 / inf_stats['median']:.1f} FPS")
    print(f"Throughput (mean):   {1000.0 / inf_stats['mean']:.1f} FPS")
    print()
    
    print("Component breakdown (by median time):")
    sorted_components = sorted(summary['components'].items(), 
                             key=lambda x: x[1]['median'], reverse=True)
    
    for component, stats in sorted_components:
        percentage = stats.get('percentage', 0)
        median_percentage = (stats['median'] / inf_stats['median']) * 100 if inf_stats['median'] > 0 else 0
        sample_count = stats.get('sample_count', 0)
        print(f"  {component:20}: {stats['median']:8.3f} ms ({median_percentage:5.1f}%) [{sample_count:3d} samples]")
    
    print(f"\nHalf-th Epoch (Median) Summary:")
    print(f"  Overall median time: {inf_stats['median']:.3f} ms")
    print(f"  Median throughput:   {1000.0 / inf_stats['median']:.1f} FPS")
    print(f"  Component median times:")
    for component, stats in sorted_components:
        sample_count = stats.get('sample_count', 0)
        print(f"    {component:20}: {stats['median']:8.3f} ms [{sample_count:3d} samples]")
    
    print(f"\nResults saved to:")
    print(f"  Detailed: {results_file}")
    print(f"  Summary:  {summary_file}")
    print(f"  CSV:      {csv_file}")

# =============================================================================
# PyTorch Profiler and Power Monitoring
# =============================================================================
def profile_model_pytorch(model, device, output_dir, input_shape=(1, 3, 224, 224), 
                          warmup_runs=20, profile_runs=100, trace_name="model_profile"):
    """
    Profile model with PyTorch Profiler + GPU power monitoring (streamlined)
    
    Args:
        model: The model to profile
        device: Device to run on  
        output_dir: Directory to save results
        input_shape: Input tensor shape (default: (1, 3, 224, 224))
        warmup_runs: Warmup iterations (default: 20)
        profile_runs: Profile iterations (default: 100)
        trace_name: Output filename prefix
    
    Returns:
        dict: Results with median timing + power/energy metrics
    """
    
    # Setup GPU power monitoring
    power_monitor = None
    try:
        import pynvml
        import os
        pynvml.nvmlInit()
        
        # Handle CUDA_VISIBLE_DEVICES mapping
        pytorch_gpu_index = int(str(device).split(':')[-1]) if ':' in str(device) else 0
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        
        if cuda_visible_devices is not None:
            visible_gpus = [int(x.strip()) for x in cuda_visible_devices.split(',') if x.strip().isdigit()]
            actual_gpu_index = visible_gpus[pytorch_gpu_index] if pytorch_gpu_index < len(visible_gpus) else pytorch_gpu_index
        else:
            actual_gpu_index = pytorch_gpu_index
            
        handle = pynvml.nvmlDeviceGetHandleByIndex(actual_gpu_index)
        power_monitor = handle
        print(f"GPU power monitoring enabled (GPU {actual_gpu_index})")
        
    except Exception as e:
        print(f"Power monitoring unavailable: {e}")
        print("Install: pip install nvidia-ml-py")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Display profiling configuration
    print(f"Input shape: {input_shape}")
    print(f"Profiling: {warmup_runs} warmup + {profile_runs} runs")
    
    # Warmup runs
    print(f"Warming up with {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Simple profiler schedule
    profiler_schedule = schedule(skip_first=0, wait=0, warmup=0, active=profile_runs, repeat=1)
    
    # Power monitoring storage
    power_readings = []
    inference_times = []
    
    def collect_power():
        """Collect GPU power reading"""
        if power_monitor:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(power_monitor)
                power_w = power_mw / 1000.0
                power_readings.append(power_w)
            except:
                pass
    
    print(f"Starting PyTorch Profiler with {profile_runs} runs...")
    
    # Configure profiler activities
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    # Start profiling with PyTorch Profiler (no TensorBoard for speed)
    with profile(
        activities=activities,
        schedule=profiler_schedule,
        # on_trace_ready=tensorboard_trace_handler(str(output_path)),  # Disabled for speed
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        
        with torch.no_grad():
            for i in range(profile_runs):
                # Collect power before inference
                collect_power()
                
                # Time the inference
                torch.cuda.synchronize()
                start_time = time.time()
                output = model(dummy_input)
                torch.cuda.synchronize()
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000)
                
                # Collect power after inference
                for _ in range(3):
                    collect_power()
                
                prof.step()
                
                if (i + 1) % 20 == 0:
                    median = statistics.median(inference_times)
                    print(f"  {i + 1}/{profile_runs} runs (median: {median:.2f}ms, power samples: {len(power_readings)})")
    
    # Calculate timing statistics (median-focused)
    timing_stats = {
        'median_ms': statistics.median(inference_times),
        'mean_ms': statistics.mean(inference_times),
        'std_ms': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
        'min_ms': min(inference_times),
        'max_ms': max(inference_times),
    }
    
    # Add quartiles if enough data
    if len(inference_times) >= 4:
        quartiles = statistics.quantiles(inference_times, n=4)
        timing_stats.update({
            'q25_ms': quartiles[0],
            'q75_ms': quartiles[2], 
            'iqr_ms': quartiles[2] - quartiles[0]
        })
    
    # Calculate power and energy statistics
    power_stats = {}
    energy_stats = {}
    
    if power_readings:
        power_stats = {
            'median_watts': statistics.median(power_readings),
            'mean_watts': statistics.mean(power_readings),
            'std_watts': statistics.stdev(power_readings) if len(power_readings) > 1 else 0,
            'min_watts': min(power_readings),
            'max_watts': max(power_readings),
        }
        
        # Energy calculation (using median values)
        median_power = power_stats['median_watts']
        median_time_s = timing_stats['median_ms'] / 1000.0
        energy_j = median_power * median_time_s
        energy_wh = energy_j / 3600.0
        
        energy_stats = {
            'energy_per_inference_joules': energy_j,
            'energy_per_inference_wh': energy_wh,
            'energy_per_inference_mwh': energy_wh * 1000
        }
    
    # Export Chrome trace (lightweight version)
    trace_file = output_path / f"{trace_name}.json"
    try:
        prof.export_chrome_trace(str(trace_file))
        print(f"Chrome trace exported to: {trace_file}")
    except Exception as e:
        print(f"Warning: Chrome trace export failed: {e}")
        trace_file = "Not generated"
    
    # Export profiler table
    table_file = output_path / f"{trace_name}_table.txt"
    with open(table_file, 'w') as f:
        f.write("=== PyTorch Profiler Key Averages ===\n\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        f.write("\n\n=== PyTorch Profiler Key Averages by Input Shape ===\n\n")
        f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))
        f.write("\n\n=== PyTorch Profiler Key Averages by Stack ===\n\n")
        f.write(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20))
    
    # Compile results
    results = {
        'config': {
            'model_name': model.__class__.__name__,
            'input_shape': input_shape,
            'warmup_runs': warmup_runs,
            'profile_runs': profile_runs,
            'device': str(device)
        },
        'timing': timing_stats,
        'power': power_stats,
        'energy': energy_stats,
        'throughput': {
            'fps_median': 1000.0 / timing_stats['median_ms'],
            'fps_mean': 1000.0 / timing_stats['mean_ms'],
            'samples_per_second_median': 1000.0 / timing_stats['median_ms']
        },
        'files': {
            'chrome_trace': str(trace_file),
            'profiler_table': str(table_file),
            'results_json': str(output_path / f"{trace_name}_results.json")
        }
    }
    
    # Save comprehensive results
    results_file = output_path / f"{trace_name}_results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== Profiling Results ===")
    print(f"Mean time:     {timing_stats['mean_ms']:.3f} ms")
    print(f"Median time:   {timing_stats['median_ms']:.3f} ms")
    print(f"Throughput:    {results['throughput']['fps_mean']:.1f} FPS")
    
    if power_stats:
        print(f"GPU mean power:   {power_stats['mean_watts']:.2f} W")
        print(f"GPU median power: {power_stats['median_watts']:.2f} W")
        print(f"Energy:           {energy_stats['energy_per_inference_mwh']:.3f} mWh/inference")
    
    print(f"Results: {results_file}")
    print(f"Table:   {table_file}")
    print(f"Trace:   {trace_file}")
    
    return results


# =============================================================================
# CPU Profiling and Power Monitoring
# =============================================================================
class CPUMonitor:
    """
    Unified CPU monitoring class that estimates power consumption without RAPL.
    Uses CPU usage, frequency, and system metrics to estimate power consumption.
    """
    
    def __init__(self, monitor_interval: float = 0.1, cpu_base_power: float = 15.0, cpu_max_power: float = None):
        self.monitor_interval = monitor_interval
        self.cpu_base_power = cpu_base_power
        self.monitoring = False
        self.monitor_data = []
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_threads = psutil.cpu_count(logical=True)
        self.cpu_cores = self.cpu_count
        self.process = psutil.Process()
        self.process.cpu_percent()  # Initialize counter for first call to return meaningful value
        
        # Try to detect CPU type for better max power estimation
        if cpu_max_power is not None:
            self.cpu_max_power = cpu_max_power
        else:
            self.cpu_max_power = 65.0  # Default fallback
            try:
                # AMD Ryzen Threadripper PRO 3955WX detection
                cpu_info = ""
                if hasattr(platform, 'processor'):
                    cpu_info = platform.processor()
                
                if not cpu_info:
                    try:
                        cpu_info = subprocess.check_output(['lscpu']).decode()
                    except:
                        pass
                
                if "3955WX" in cpu_info or "Threadripper" in cpu_info:
                    self.cpu_max_power = 280.0
                elif self.cpu_threads >= 64:
                    self.cpu_max_power = 250.0
                elif self.cpu_threads >= 32:
                    self.cpu_max_power = 150.0
            except Exception as e:
                print(f"Warning: Could not auto-detect CPU power limit: {e}")
    
    def get_cpu_frequency(self) -> float:
        """Get current CPU frequency in MHz."""
        try:
            # Try to read from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'cpu MHz' in line:
                        return float(line.split(':')[1].strip())
        except:
            pass
        
        # Fallback: estimate based on CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return 2000.0 + (cpu_percent / 100.0) * 1000.0  # Estimate between 2-3 GHz
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get detailed CPU usage statistics."""
        # Get overall usage first
        cpu_percent = psutil.cpu_percent(interval=self.monitor_interval)
        
        # Get process-specific usage
        try:
            process_cpu_percent = self.process.cpu_percent()
        except:
            process_cpu_percent = 0.0
        
        # Get frequency information
        cpu_freq = psutil.cpu_freq()
        
        return {
            'cpu_percent': cpu_percent,
            'process_cpu_percent': process_cpu_percent,
            'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
            'cpu_max_freq_mhz': cpu_freq.max if cpu_freq and cpu_freq.max > 0 else 4000.0,
            'cpu_count': self.cpu_threads
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'swap_percent': swap.percent,
            'swap_used_gb': swap.used / (1024**3)
        }
    
    def get_system_load(self) -> Dict[str, float]:
        """Get system load statistics."""
        load_avg = psutil.getloadavg()
        
        return {
            'load_1min': load_avg[0],
            'load_5min': load_avg[1],
            'load_15min': load_avg[2]
        }
    
    def estimate_power_consumption(self, cpu_usage: Dict[str, float]) -> float:
        """Estimate power consumption based on CPU usage and frequency."""
        cpu_percent = cpu_usage['cpu_percent']
        process_cpu_percent = cpu_usage.get('process_cpu_percent', 0.0)
        
        # Get frequency, default to a reasonable value if not available
        cpu_freq = cpu_usage.get('cpu_freq_mhz', 0)
        max_freq = cpu_usage.get('cpu_max_freq_mhz', 3900.0)
        
        if cpu_freq <= 0:
            cpu_freq = max_freq * 0.75
        
        # Base power consumption (idle)
        # For 3955WX, idle is around 35W
        base_power = 35.0
        
        # Frequency scaling factor
        freq_factor = (cpu_freq / max_freq) ** 1.5
        
        # Total system usage factor
        system_usage_factor = cpu_percent / 100.0
        
        # Estimated total system power
        # We use a sub-linear factor for high-core count CPUs
        total_power = base_power + (self.cpu_max_power - base_power) * freq_factor * (system_usage_factor ** 0.6)
        
        # Now estimate power specifically for the inference process
        # We calculate the incremental power added by the process
        # A process usage of 100% means it's using 1 thread fully.
        # On a multi-threaded system, process_cpu_percent can exceed 100%.
        # We normalize it to the total system threads.
        process_usage_factor = process_cpu_percent / (self.cpu_threads * 100.0)
        
        # Inference power is estimated as the share of active power (Total - Base) 
        # that is attributable to this process.
        active_power = total_power - base_power
        if cpu_percent > 0:
            # psutil.cpu_percent is 0-100 (average), so we multiply by threads to get total system capacity
            system_total_percent = cpu_percent * self.cpu_threads
            inference_power = active_power * (process_cpu_percent / system_total_percent)
        else:
            inference_power = 0.0
            
        # Store both in the sample later, but return total_power for compatibility
        cpu_usage['inference_power_w'] = inference_power
        return min(total_power, self.cpu_max_power)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            timestamp = time.time()
            
            # Collect all metrics
            cpu_usage = self.get_cpu_usage()
            memory_usage = self.get_memory_usage()
            system_load = self.get_system_load()
            
            # Estimate power consumption (this updates cpu_usage with inference_power_w)
            total_power = self.estimate_power_consumption(cpu_usage)
            
            # Combine all data
            sample = {
                'timestamp': timestamp,
                'estimated_power_w': total_power,
                'inference_power_w': cpu_usage.get('inference_power_w', 0.0),
                **cpu_usage,
                **memory_usage,
                **system_load
            }
            
            self.monitor_data.append(sample)
            time.sleep(self.monitor_interval)
    
    def start_monitoring(self):
        """Start monitoring."""
        self.monitoring = True
        self.monitor_data = []
        self.start_time = time.time()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        self.end_time = time.time()
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        """Calculate statistics from collected data."""
        if not self.monitor_data:
            return {'error': 'No monitoring data collected'}
        
        duration = self.end_time - self.start_time if self.end_time else 0
        
        # Extract power data
        power_data = [d['estimated_power_w'] for d in self.monitor_data]
        inference_power_data = [d['inference_power_w'] for d in self.monitor_data]
        cpu_percent_data = [d['cpu_percent'] for d in self.monitor_data]
        process_cpu_data = [d['process_cpu_percent'] for d in self.monitor_data]
        memory_percent_data = [d['memory_percent'] for d in self.monitor_data]
        
        # Calculate statistics
        stats = {
            'device': 'cpu_estimated',
            'duration_s': duration,
            'samples': len(self.monitor_data),
            'monitoring_method': 'estimated_power',
            'avg_power_w': sum(power_data) / len(power_data) if power_data else 0,
            'median_power_w': statistics.median(power_data) if power_data else 0,
            'max_power_w': max(power_data) if power_data else 0,
            'min_power_w': min(power_data) if power_data else 0,
            'avg_inference_power_w': sum(inference_power_data) / len(inference_power_data) if inference_power_data else 0,
            'median_inference_power_w': statistics.median(inference_power_data) if inference_power_data else 0,
            'max_inference_power_w': max(inference_power_data) if inference_power_data else 0,
            'total_energy_j': sum(power_data) * self.monitor_interval if power_data else 0,
            'inference_energy_j': sum(inference_power_data) * self.monitor_interval if inference_power_data else 0,
            'avg_cpu_usage_percent': sum(cpu_percent_data) / len(cpu_percent_data) if cpu_percent_data else 0,
            'avg_process_cpu_percent': sum(process_cpu_data) / len(process_cpu_data) if process_cpu_data else 0,
            'max_cpu_usage_percent': max(cpu_percent_data) if cpu_percent_data else 0,
            'avg_memory_usage_percent': sum(memory_percent_data) / len(memory_percent_data) if memory_percent_data else 0,
            'cpu_cores': self.cpu_cores,
            'cpu_threads': self.cpu_threads,
            'power_data': power_data,
            'inference_power_data': inference_power_data,
            'cpu_usage_data': cpu_percent_data,
            'memory_usage_data': memory_percent_data
        }
        
        return stats


def monitor_cpu_power(model, input_tensor, warmup_runs=3, test_runs=10, monitor_interval=0.1, 
                     cpu_base_power=15.0, cpu_max_power=None):
    """
    Monitor CPU power using alternative methods.
    
    Args:
        model: PyTorch model to profile
        input_tensor: Input tensor for inference
        warmup_runs: Number of warmup runs
        test_runs: Number of test runs
        monitor_interval: Monitoring interval in seconds
        cpu_base_power: Base CPU power consumption in watts
        cpu_max_power: Maximum CPU power consumption in watts
    
    Returns:
        Dict: Statistics including power consumption and inference performance
    """
    model.eval()
    
    # Warmup
    print(f"Warming up with {warmup_runs} runs...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Initialize monitor
    monitor = CPUMonitor(monitor_interval=monitor_interval, 
                        cpu_base_power=cpu_base_power, 
                        cpu_max_power=cpu_max_power)
    
    print("Starting CPU power monitoring...")
    monitor.start_monitoring()
    
    # Run inference
    inference_times = []
    for i in range(test_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        end_time = time.time()
        inference_times.append((end_time - start_time) * 1000)
    
    # Stop monitoring
    power_stats = monitor.stop_monitoring()
    
    # Add inference data
    power_stats['inference_times_ms'] = inference_times
    power_stats['avg_inference_time_ms'] = sum(inference_times) / len(inference_times)
    power_stats['median_inference_time_ms'] = statistics.median(inference_times) if inference_times else 0
    power_stats['throughput_fps'] = 1000.0 / (sum(inference_times) / len(inference_times))
    
    return power_stats


def save_cpu_power_report(stats: Dict, output_path: str):
    """Save CPU power monitoring report."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("CPU Power Monitoring Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Duration: {stats.get('duration_s', 0):.2f} seconds\n")
        f.write(f"Samples: {stats.get('samples', 0)}\n")
        f.write(f"Method: {stats.get('monitoring_method', 'unknown')}\n")
        
        f.write(f"\nEstimated Total System Power:\n")
        f.write(f"  Average: {stats.get('avg_power_w', 0):.2f} W\n")
        f.write(f"  Maximum: {stats.get('max_power_w', 0):.2f} W\n")
        f.write(f"  Minimum: {stats.get('min_power_w', 0):.2f} W\n")
        f.write(f"  Total Energy: {stats.get('total_energy_j', 0):.4f} J\n")
        
        f.write(f"\nEstimated Inference-Specific Power (Incremental):\n")
        f.write(f"  Average: {stats.get('avg_inference_power_w', 0):.2f} W\n")
        f.write(f"  Median:  {stats.get('median_inference_power_w', 0):.2f} W\n")
        f.write(f"  Maximum: {stats.get('max_inference_power_w', 0):.2f} W\n")
        f.write(f"  Inference Energy: {stats.get('inference_energy_j', 0):.4f} J\n")
        
        f.write(f"\nCPU Usage:\n")
        f.write(f"  System Average: {stats.get('avg_cpu_usage_percent', 0):.1f}%\n")
        f.write(f"  Process Average: {stats.get('avg_process_cpu_percent', 0):.1f}%\n")
        f.write(f"  Maximum: {stats.get('max_cpu_usage_percent', 0):.1f}%\n")
        
        f.write(f"\nMemory Usage:\n")
        f.write(f"  Average: {stats.get('avg_memory_usage_percent', 0):.1f}%\n")
        
        f.write(f"\nSystem Information:\n")
        f.write(f"  CPU Cores: {stats.get('cpu_cores', 0)}\n")
        f.write(f"  CPU Threads: {stats.get('cpu_threads', 0)}\n")
        
        f.write(f"\nInference Performance:\n")
        f.write(f"  Average Time: {stats.get('avg_inference_time_ms', 0):.2f} ms\n")
        f.write(f"  Throughput: {stats.get('throughput_fps', 0):.2f} FPS\n")
        
        # Energy efficiency
        total_energy = stats.get('total_energy_j', 0)
        if total_energy > 0:
            energy_per_inference = total_energy / stats.get('samples', 1)
            f.write(f"\nEnergy Efficiency:\n")
            f.write(f"  Total Energy: {total_energy:.4f} J\n")
            f.write(f"  Energy per Sample: {energy_per_inference:.4f} J\n")
            f.write(f"  Energy per Inference: {energy_per_inference:.4f} J\n")
    
    # Save detailed JSON
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"CPU power report saved to: {output_file}")
    print(f"JSON data saved to: {json_file}")


def profile_cpu_model(model, device, input_shape=(1, 3, 224, 224), warmup_runs=3, test_runs=10, 
                     output_file=None, monitor_interval=0.1, cpu_base_power=15.0, cpu_max_power=None):
    """
    Comprehensive CPU model profiling with power monitoring.
    
    Args:
        model: PyTorch model to profile
        device: Device to run on (should be CPU)
        input_shape: Input tensor shape
        warmup_runs: Number of warmup runs
        test_runs: Number of test runs
        output_file: Output file path for report
        monitor_interval: Monitoring interval in seconds
        cpu_base_power: Base CPU power consumption in watts
        cpu_max_power: Maximum CPU power consumption in watts
    
    Returns:
        Dict: Comprehensive profiling results
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    print("Starting CPU model profiling with power monitoring...")
    power_stats = monitor_cpu_power(
        model, 
        dummy_input, 
        warmup_runs=warmup_runs,
        test_runs=test_runs,
        monitor_interval=monitor_interval,
        cpu_base_power=cpu_base_power,
        cpu_max_power=cpu_max_power
    )
    
    # Display results
    print(f"\nCPU Model Performance:")
    print(f"  Mean inference time: {power_stats.get('avg_inference_time_ms', 0):.3f} ms")
    print(f"  Median inference time: {power_stats.get('median_inference_time_ms', 0):.3f} ms")
    print(f"  Throughput: {power_stats.get('throughput_fps', 0):.1f} FPS")
    print(f"  Estimated System Median Power: {power_stats.get('median_power_w', 0):.2f} W")
    print(f"  Estimated Inference Median Power: {power_stats.get('median_inference_power_w', 0):.2f} W")
    print(f"  Estimated Inference Energy: {power_stats.get('inference_energy_j', 0):.4f} J")
    
    if 'avg_cpu_usage_percent' in power_stats:
        print(f"  Average System CPU Usage: {power_stats['avg_cpu_usage_percent']:.1f}%")
        print(f"  Average Process CPU Usage: {power_stats['avg_process_cpu_percent']:.1f}%")
    
    # Save report if output file specified
    if output_file:
        save_cpu_power_report(power_stats, output_file)
    
    return power_stats
