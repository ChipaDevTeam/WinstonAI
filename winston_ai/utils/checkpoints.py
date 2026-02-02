"""
Model checkpoint management utilities
"""

import torch
import os
import glob
import re
from typing import Optional, Dict, Any


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    episode: int,
    metrics: Optional[Dict[str, Any]],
    filepath: str
):
    """
    Save model checkpoint
    
    Args:
        model: The model to save
        optimizer: Optional optimizer state
        episode: Current training episode
        metrics: Optional training metrics
        filepath: Path where to save the checkpoint
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'episode': episode,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, filepath)
    print(f"[Checkpoint] Saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to the checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        device: Optional device to load tensors to
        
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Checkpoint] Loaded model state from {filepath}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[Checkpoint] Loaded optimizer state from {filepath}")
    
    return checkpoint


def find_latest_checkpoint(directory: str, pattern: str = "*.pth") -> Optional[str]:
    """
    Find the latest checkpoint file in a directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    checkpoint_files = glob.glob(os.path.join(directory, pattern))
    
    if not checkpoint_files:
        return None
    
    # Try to extract episode numbers from filenames
    def extract_episode(filename):
        match = re.search(r'episode[_-]?(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0
    
    # Sort by episode number
    checkpoint_files.sort(key=extract_episode, reverse=True)
    
    return checkpoint_files[0]


def list_checkpoints(directory: str, pattern: str = "*.pth") -> list:
    """
    List all checkpoint files in a directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of checkpoint file paths
    """
    checkpoint_files = glob.glob(os.path.join(directory, pattern))
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    return checkpoint_files
