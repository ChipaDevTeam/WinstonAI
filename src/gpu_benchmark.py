"""
GPU Performance Benchmark for WinstonAI
Compare original vs GPU-optimized model performance and resource utilization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import psutil
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import original model if available
try:
    from train import WinstonAI as OriginalWinstonAI
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    print("Original WinstonAI not available for comparison")

# Import GPU-optimized model
from train_gpu_optimized import AdvancedWinstonAI

# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0

def benchmark_model(model_class, model_name, state_size, action_size, batch_sizes, num_iterations=100):
    """Benchmark a specific model"""
    print(f"\n{'='*50}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*50}")
    
    results = {
        'model_name': model_name,
        'batch_results': {},
        'model_parameters': 0,
        'model_size_mb': 0
    }
    
    try:
        # Create model
        if model_name == "Original WinstonAI":
            model = model_class(state_size, action_size, hidden_size=512).to(device)
        else:
            model = model_class(state_size, action_size, hidden_size=4096).to(device)
        
        model.eval()
        
        # Calculate model parameters and size
        total_params = sum(p.numel() for p in model.parameters())
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        results['model_parameters'] = total_params
        results['model_size_mb'] = model_size
        
        print(f"Model Parameters: {total_params:,}")
        print(f"Model Size: {model_size:.2f} MB")
        
        # Benchmark different batch sizes
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            try:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate test data
                test_data = torch.randn(batch_size, state_size).to(device)
                
                # Warm up
                for _ in range(10):
                    with torch.no_grad():
                        _ = model(test_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Benchmark forward pass
                start_time = time.time()
                memory_before = get_gpu_memory_usage()
                
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = model(test_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                memory_after = get_gpu_memory_usage()
                
                # Calculate metrics
                total_time = end_time - start_time
                avg_time_per_batch = total_time / num_iterations
                throughput = batch_size / avg_time_per_batch
                memory_used = memory_after[0] - memory_before[0]
                
                batch_result = {
                    'batch_size': batch_size,
                    'total_time': total_time,
                    'avg_time_per_batch': avg_time_per_batch,
                    'throughput_samples_per_sec': throughput,
                    'gpu_memory_used_gb': memory_used,
                    'gpu_memory_reserved_gb': memory_after[1]
                }
                
                results['batch_results'][batch_size] = batch_result
                
                print(f"  Avg time per batch: {avg_time_per_batch*1000:.2f} ms")
                print(f"  Throughput: {throughput:.1f} samples/sec")
                print(f"  GPU Memory Used: {memory_used:.2f} GB")
                print(f"  GPU Memory Reserved: {memory_after[1]:.2f} GB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM Error at batch size {batch_size}")
                    break
                else:
                    raise e
    
    except Exception as e:
        print(f"Error benchmarking {model_name}: {e}")
        return None
    
    return results

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing models"""
    print("WinstonAI GPU Optimization Benchmark")
    print("=" * 60)
    
    # Configuration
    state_size = 2000  # 20 features * 100 lookback
    action_size = 3
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Progressive batch sizes
    num_iterations = 50
    
    all_results = {}
    
    # Benchmark Original WinstonAI (if available)
    if ORIGINAL_AVAILABLE:
        original_results = benchmark_model(
            OriginalWinstonAI, 
            "Original WinstonAI", 
            state_size, 
            action_size, 
            batch_sizes[:6],  # Smaller batch sizes for original
            num_iterations
        )
        if original_results:
            all_results['original'] = original_results
    
    # Benchmark GPU-Optimized WinstonAI
    optimized_results = benchmark_model(
        AdvancedWinstonAI,
        "GPU-Optimized WinstonAI",
        state_size,
        action_size,
        batch_sizes,
        num_iterations
    )
    if optimized_results:
        all_results['optimized'] = optimized_results
    
    return all_results

def plot_benchmark_results(results):
    """Plot comprehensive benchmark comparison"""
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for plotting
    models = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot 1: Throughput vs Batch Size
    ax1 = axes[0, 0]
    for i, (model_key, model_data) in enumerate(results.items()):
        batch_sizes = []
        throughputs = []
        
        for batch_size, batch_result in model_data['batch_results'].items():
            batch_sizes.append(batch_size)
            throughputs.append(batch_result['throughput_samples_per_sec'])
        
        ax1.plot(batch_sizes, throughputs, 'o-', color=colors[i], 
                label=model_data['model_name'], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (samples/sec)')
    ax1.set_title('Throughput vs Batch Size', fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: GPU Memory Usage vs Batch Size
    ax2 = axes[0, 1]
    for i, (model_key, model_data) in enumerate(results.items()):
        batch_sizes = []
        memory_usage = []
        
        for batch_size, batch_result in model_data['batch_results'].items():
            batch_sizes.append(batch_size)
            memory_usage.append(batch_result['gpu_memory_used_gb'])
        
        ax2.plot(batch_sizes, memory_usage, 'o-', color=colors[i],
                label=model_data['model_name'], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('GPU Memory Used (GB)')
    ax2.set_title('GPU Memory Usage vs Batch Size', fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Model Comparison (Parameters and Size)
    ax3 = axes[1, 0]
    model_names = [data['model_name'] for data in results.values()]
    model_params = [data['model_parameters'] for data in results.values()]
    model_sizes = [data['model_size_mb'] for data in results.values()]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    ax3_twin = ax3.twinx()
    bars1 = ax3.bar(x_pos - width/2, [p/1e6 for p in model_params], width, 
                   label='Parameters (M)', color='skyblue', alpha=0.8)
    bars2 = ax3_twin.bar(x_pos + width/2, model_sizes, width,
                        label='Size (MB)', color='lightcoral', alpha=0.8)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Parameters (Millions)', color='blue')
    ax3_twin.set_ylabel('Model Size (MB)', color='red')
    ax3.set_title('Model Complexity Comparison', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=15)
    
    # Add value labels on bars
    for bar, value in zip(bars1, model_params):
        height = bar.get_height()
        ax3.annotate(f'{value/1e6:.1f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar, value in zip(bars2, model_sizes):
        height = bar.get_height()
        ax3_twin.annotate(f'{value:.1f}MB',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Performance Summary Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for model_key, model_data in results.items():
        # Get best throughput
        best_throughput = 0
        best_batch_size = 0
        max_memory = 0
        
        for batch_size, batch_result in model_data['batch_results'].items():
            throughput = batch_result['throughput_samples_per_sec']
            memory = batch_result['gpu_memory_used_gb']
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
            
            if memory > max_memory:
                max_memory = memory
        
        table_data.append([
            model_data['model_name'],
            f"{model_data['model_parameters']/1e6:.1f}M",
            f"{best_throughput:.0f}",
            f"{best_batch_size}",
            f"{max_memory:.2f}GB"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'Parameters', 'Max Throughput\n(samples/sec)', 
                               'Best Batch Size', 'Max GPU Memory'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title('Performance Summary', fontweight='bold', pad=20)
    
    plt.suptitle('WinstonAI GPU Optimization Benchmark Results', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'winston_ai_gpu_benchmark_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nBenchmark plots saved as: {filename}")
    
    plt.show()

def save_benchmark_results(results):
    """Save benchmark results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'winston_ai_gpu_benchmark_{timestamp}.json'
    
    # Convert results to JSON-serializable format
    json_results = {}
    for model_key, model_data in results.items():
        json_results[model_key] = {
            'model_name': model_data['model_name'],
            'model_parameters': model_data['model_parameters'],
            'model_size_mb': model_data['model_size_mb'],
            'batch_results': model_data['batch_results']
        }
    
    # Add system information
    json_results['system_info'] = {
        'timestamp': timestamp,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
        'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
        'cpu_count': psutil.cpu_count(),
        'ram_gb': psutil.virtual_memory().total / 1024**3,
        'python_version': sys.version,
        'pytorch_version': torch.__version__
    }
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Benchmark results saved as: {filename}")

def print_performance_improvements(results):
    """Print performance improvement summary"""
    if len(results) < 2:
        print("Need at least 2 models for comparison")
        return
    
    print("\n" + "="*60)
    print("PERFORMANCE IMPROVEMENT SUMMARY")
    print("="*60)
    
    # Get original and optimized results
    original = results.get('original')
    optimized = results.get('optimized')
    
    if not original or not optimized:
        print("Cannot compare - missing original or optimized results")
        return
    
    # Compare model complexity
    param_improvement = optimized['model_parameters'] / original['model_parameters']
    size_improvement = optimized['model_size_mb'] / original['model_size_mb']
    
    print(f"Model Complexity:")
    print(f"  Parameters: {param_improvement:.1f}x increase ({original['model_parameters']:,} → {optimized['model_parameters']:,})")
    print(f"  Model Size: {size_improvement:.1f}x increase ({original['model_size_mb']:.1f}MB → {optimized['model_size_mb']:.1f}MB)")
    
    # Compare throughput at similar batch sizes
    common_batch_sizes = set(original['batch_results'].keys()) & set(optimized['batch_results'].keys())
    
    if common_batch_sizes:
        print(f"\nThroughput Comparison (common batch sizes):")
        for batch_size in sorted(common_batch_sizes):
            orig_throughput = original['batch_results'][batch_size]['throughput_samples_per_sec']
            opt_throughput = optimized['batch_results'][batch_size]['throughput_samples_per_sec']
            improvement = opt_throughput / orig_throughput
            
            print(f"  Batch {batch_size:3d}: {improvement:.2f}x ({orig_throughput:.1f} → {opt_throughput:.1f} samples/sec)")
    
    # GPU utilization improvement
    max_orig_memory = max(result['gpu_memory_used_gb'] for result in original['batch_results'].values())
    max_opt_memory = max(result['gpu_memory_used_gb'] for result in optimized['batch_results'].values())
    memory_improvement = max_opt_memory / max_orig_memory if max_orig_memory > 0 else float('inf')
    
    print(f"\nGPU Memory Utilization:")
    print(f"  Maximum Usage: {memory_improvement:.1f}x increase ({max_orig_memory:.2f}GB → {max_opt_memory:.2f}GB)")
    
    # Maximum achievable batch size
    max_orig_batch = max(original['batch_results'].keys())
    max_opt_batch = max(optimized['batch_results'].keys())
    batch_improvement = max_opt_batch / max_orig_batch
    
    print(f"\nMaximum Batch Size:")
    print(f"  Capacity: {batch_improvement:.1f}x increase ({max_orig_batch} → {max_opt_batch})")
    
    print("="*60)

if __name__ == "__main__":
    print("Starting WinstonAI GPU Optimization Benchmark...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
    
    # Run benchmark
    results = run_comprehensive_benchmark()
    
    if results:
        # Save results
        save_benchmark_results(results)
        
        # Print improvement summary
        print_performance_improvements(results)
        
        # Plot results
        plot_benchmark_results(results)
        
        print("\n[OK] Benchmark completed successfully!")
        print("[STATS] Check the generated PNG and JSON files for detailed results")
    else:
        print("[ERROR] Benchmark failed - no results generated")
