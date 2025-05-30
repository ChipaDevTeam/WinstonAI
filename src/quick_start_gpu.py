"""
Quick Start Script for WinstonAI GPU-Optimized Training
Automatically sets up and runs the high-performance training with optimal GPU utilization
"""

import os
import sys
import subprocess
import json
import torch
from datetime import datetime

def check_gpu_setup():
    """Check GPU setup and configuration"""
    print("[SEARCH] Checking GPU Setup...")
    print("=" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'[OK] Yes' if cuda_available else '[ERROR] No'}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check current GPU memory
        current_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        print(f"Current Memory Usage: {current_memory:.2f} GB allocated, {reserved_memory:.2f} GB reserved")
        
        return True
    else:
        print("[WARNING] No CUDA GPU detected. Training will run on CPU (much slower)")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing Required Packages...")
    print("=" * 50)
    
    required_packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "ta",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package} already installed")
        except ImportError:
            print(f"📥 Installing {package}...")
            if package in ["torch", "torchvision", "torchaudio"]:
                # Install PyTorch with CUDA support
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu118"
                ])
                break
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_data_file():
    """Check if data file exists"""
    data_path = r"c:\Users\tp\ComunityPrograms\all_assets_candles.csv"
    
    print(f"\n[STATS] Checking Data File...")
    print("=" * 50)
    
    if os.path.exists(data_path):
        print(f"[OK] Data file found: {data_path}")
        
        # Get file size
        file_size = os.path.getsize(data_path) / 1024**2
        print(f"📏 File size: {file_size:.2f} MB")
        
        return True
    else:
        print(f"[ERROR] Data file not found: {data_path}")
        print("Please ensure the data file exists before training")
        return False

def setup_gpu_optimization():
    """Set up optimal GPU configuration"""
    print("\n⚙️ Setting up GPU Optimization...")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("[OK] CUDNN benchmark enabled")
        print("[OK] TF32 enabled for faster training")
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.95)
        print("[OK] GPU memory fraction set to 95%")
        
        return True
    
    return False

def create_optimized_config():
    """Create optimized configuration based on GPU"""
    print("\n📝 Creating Optimized Configuration...")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Adjust configuration based on GPU memory
        if gpu_memory >= 12:  # 12GB+ (like RTX 3060 Ti)
            config = {
                "model": {
                    "hidden_size": 4096,
                    "attention_heads": 16,
                    "lstm_layers": 4,
                    "lstm_hidden_size": 2048
                },
                "training": {
                    "batch_size": 512,
                    "memory_buffer_size": 1000000,
                    "episodes": 5000,
                    "learning_rate": 0.0001
                },
                "optimization": "ultra_high"
            }
            print(f"[OK] Ultra-high performance config for {gpu_memory:.1f}GB GPU")
            
        elif gpu_memory >= 8:  # 8-12GB
            config = {
                "model": {
                    "hidden_size": 2048,
                    "attention_heads": 8,
                    "lstm_layers": 3,
                    "lstm_hidden_size": 1024
                },
                "training": {
                    "batch_size": 256,
                    "memory_buffer_size": 500000,
                    "episodes": 3000,
                    "learning_rate": 0.0001
                },
                "optimization": "high"
            }
            print(f"[OK] High performance config for {gpu_memory:.1f}GB GPU")
            
        else:  # Less than 8GB
            config = {
                "model": {
                    "hidden_size": 1024,
                    "attention_heads": 4,
                    "lstm_layers": 2,
                    "lstm_hidden_size": 512
                },
                "training": {
                    "batch_size": 128,
                    "memory_buffer_size": 100000,
                    "episodes": 2000,
                    "learning_rate": 0.0001
                },
                "optimization": "medium"
            }
            print(f"[OK] Medium performance config for {gpu_memory:.1f}GB GPU")
    else:
        # CPU configuration
        config = {
            "model": {
                "hidden_size": 512,
                "attention_heads": 2,
                "lstm_layers": 2,
                "lstm_hidden_size": 256
            },
            "training": {
                "batch_size": 32,
                "memory_buffer_size": 50000,
                "episodes": 1000,
                "learning_rate": 0.001
            },
            "optimization": "cpu"
        }
        print("[OK] CPU configuration (limited performance)")
    
    # Save configuration
    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("📁 Configuration saved to training_config.json")
    return config

def run_training():
    """Run the GPU-optimized training"""
    print("\n[GPU] Starting GPU-Optimized Training...")
    print("=" * 50)
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, "train_gpu_optimized.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] Training completed successfully!")
            print("\n[STATS] Training Output:")
            print(result.stdout)
        else:
            print("[ERROR] Training failed!")
            print("\n[SEARCH] Error Output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running training: {e}")
        return False
    
    return True

def run_benchmark():
    """Run performance benchmark"""
    print("\n[STATS] Running Performance Benchmark...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "gpu_benchmark.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] Benchmark completed!")
            print(result.stdout)
        else:
            print("[WARNING] Benchmark had issues:")
            print(result.stderr)
            
    except Exception as e:
        print(f"[WARNING] Could not run benchmark: {e}")

def main():
    """Main function to run the complete setup and training"""
    print("🤖 WinstonAI GPU-Optimized Training Setup")
    print("[TARGET] Maximum GPU Utilization for RTX 3060 Ti 12GB")
    print("=" * 60)
    print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check GPU setup
    gpu_available = check_gpu_setup()
    
    # Step 2: Install requirements
    install_requirements()
    
    # Step 3: Check data file
    if not check_data_file():
        print("\n[ERROR] Cannot proceed without data file")
        return
    
    # Step 4: Setup GPU optimization
    setup_gpu_optimization()
    
    # Step 5: Create optimized configuration
    config = create_optimized_config()
    
    # Step 6: Show training plan
    print("\n[INFO] Training Plan:")
    print("=" * 50)
    print(f"Optimization Level: {config['optimization']}")
    print(f"Model Hidden Size: {config['model']['hidden_size']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Episodes: {config['training']['episodes']}")
    print(f"Memory Buffer: {config['training']['memory_buffer_size']:,}")
    
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        estimated_usage = (config['training']['batch_size'] * config['model']['hidden_size']) / 1e6
        print(f"Estimated GPU Usage: ~{estimated_usage:.1f}GB of {gpu_memory:.1f}GB available")
    
    # Ask for confirmation
    user_input = input("\n[QUESTION] Do you want to proceed with training? (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        print("\n[GPU] Starting training process...")
        
        # Step 7: Run training
        success = run_training()
        
        if success:
            # Step 8: Run benchmark (optional)
            benchmark_input = input("\n[STATS] Run performance benchmark? (y/n): ").lower().strip()
            if benchmark_input in ['y', 'yes']:
                run_benchmark()
            
            print("\n🎉 Training session completed!")
            print("📁 Check the generated files:")
            print("   - winston_ai_gpu_final.pth (trained model)")
            print("   - winston_ai_gpu_training_results.png (training plots)")
            print("   - winston_ai_live_trading.log (training logs)")
        else:
            print("\n[ERROR] Training session failed")
    else:
        print("\n🛑 Training cancelled by user")
    
    print(f"\n[TIME] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
