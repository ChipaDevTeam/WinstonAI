"""
WinstonAI - GPU-optimized Reinforcement Learning Trading Library

A comprehensive library for training and deploying AI trading models
for binary options and forex markets.
"""

__version__ = "1.2.0"
__author__ = "ChipaDevTeam"

# Core imports for easy access
from winston_ai.models.winston_model import WinstonAI, AdvancedWinstonAI
from winston_ai.training.trainer import Trainer
from winston_ai.trading.live_trader import LiveTrader
from winston_ai.indicators.technical import TechnicalIndicators
from winston_ai.utils.config import Config

__all__ = [
    "WinstonAI",
    "AdvancedWinstonAI",
    "Trainer",
    "LiveTrader",
    "TechnicalIndicators",
    "Config",
    "__version__",
]
