"""
GPU and device management utilities for WinstonAI
"""

import torch
import warnings


def get_device():
    """
    Get the best available device (CUDA GPU if available, otherwise CPU)
    
    Returns:
        torch.device: The device to use for computations
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_gpu():
    """
    Configure GPU settings for optimal performance
    
    Returns:
        dict: GPU configuration information including device, name, and memory
    """
    device = get_device()
    
    config = {
        'device': device,
        'device_name': None,
        'total_memory_gb': None,
        'available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        config['device_name'] = torch.cuda.get_device_name()
        config['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"[GPU] Detected: {config['device_name']}")
        print(f"[GPU] Memory: {config['total_memory_gb']:.1f} GB")
        
        # Enable tensor cores and optimize memory usage
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("[WARNING] No GPU detected, falling back to CPU")
        warnings.warn("No GPU detected. Training will be significantly slower on CPU.")
    
    return config


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("[GPU Memory] No GPU available")
