"""
Configuration management for WinstonAI
"""

import json
import os
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for WinstonAI training and trading parameters
    """
    
    DEFAULT_TRAINING_CONFIG = {
        "episodes": 3000,
        "batch_size": 512,
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "target_update_frequency": 10,
        "memory_size": 100000,
        "lookback_window": 100,
        "save_frequency": 100,
    }
    
    DEFAULT_TRADING_CONFIG = {
        "max_daily_loss": 100,
        "stop_loss_percent": 0.02,
        "take_profit_percent": 0.04,
        "trade_amount": 10,
        "payout_ratio": 0.8,
        "max_trades_per_day": 50,
    }
    
    DEFAULT_GPU_CONFIG = {
        "device": "cuda",
        "mixed_precision": True,
        "gradient_checkpointing": False,
        "memory_efficient": True,
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        self.training = self.DEFAULT_TRAINING_CONFIG.copy()
        self.trading = self.DEFAULT_TRADING_CONFIG.copy()
        self.gpu = self.DEFAULT_GPU_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from a JSON file
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'training' in config:
            self.training.update(config['training'])
        if 'trading' in config:
            self.trading.update(config['trading'])
        if 'gpu' in config:
            self.gpu.update(config['gpu'])
    
    def save_to_file(self, config_path: str):
        """
        Save configuration to a JSON file
        
        Args:
            config_path: Path where to save the configuration
        """
        config = {
            'training': self.training,
            'trading': self.trading,
            'gpu': self.gpu,
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def update(self, section: str, **kwargs):
        """
        Update configuration parameters
        
        Args:
            section: Configuration section ('training', 'trading', or 'gpu')
            **kwargs: Configuration parameters to update
        """
        if section == 'training':
            self.training.update(kwargs)
        elif section == 'trading':
            self.trading.update(kwargs)
        elif section == 'gpu':
            self.gpu.update(kwargs)
        else:
            raise ValueError(f"Unknown section: {section}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        if section == 'training':
            return self.training.get(key, default)
        elif section == 'trading':
            return self.trading.get(key, default)
        elif section == 'gpu':
            return self.gpu.get(key, default)
        else:
            raise ValueError(f"Unknown section: {section}")
    
    def __repr__(self):
        return f"Config(training={self.training}, trading={self.trading}, gpu={self.gpu})"
