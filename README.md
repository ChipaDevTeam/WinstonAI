# WinstonAI 🤖📈

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**WinstonAI** is an advanced GPU-optimized Reinforcement Learning trading bot for binary options trading. Built with PyTorch and designed to fully utilize modern GPU architecture (RTX 3060 Ti 12GB VRAM), WinstonAI employs sophisticated deep learning techniques for real-time market analysis and trading decisions.

## 🌟 Key Features

- **🧠 Advanced Deep Learning Architecture**
  - Deep Reinforcement Learning with Dueling DQN
  - Multi-head attention mechanism (16 heads)
  - 4-layer LSTM networks for temporal pattern recognition
  - Mixed precision training for optimal GPU utilization

- **⚡ GPU Optimization**
  - Fully optimized for NVIDIA RTX 3060 Ti (12GB VRAM)
  - Mixed precision (FP16/FP32) training
  - Tensor core acceleration
  - Memory-efficient replay buffer management

- **📊 Advanced Technical Analysis**
  - 50+ technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, etc.)
  - Real-time market data processing
  - Multi-asset support (Forex pairs: EUR/USD, GBP/USD, USD/JPY, etc.)

- **🎯 Risk Management**
  - Configurable stop-loss and take-profit levels
  - Maximum daily loss limits
  - Position sizing algorithms
  - Real-time performance monitoring

- **🔌 Live Trading Integration**
  - Real-time integration with PocketOption API
  - Automated trade execution
  - Live market data streaming
  - Performance logging and analytics

## 📋 Requirements

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended: RTX 3060 Ti or better)
- CUDA 11.8 or higher
- 8GB+ VRAM (12GB recommended)
- 16GB+ system RAM

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ChipaDevTeam/WinstonAI.git
cd WinstonAI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install WinstonAI as a package:
```bash
pip install -e .
```

### Quick Start

Run the quick start example:
```bash
python examples/quickstart.py
```

Or train with more control:
```bash
python examples/train_model.py
```

### Using the Library

After installation, you can import WinstonAI in your Python scripts:

```python
from winston_ai import Trainer, Config, LiveTrader
from winston_ai import WinstonAI, AdvancedWinstonAI
from winston_ai.indicators import TechnicalIndicators
```

See `examples/` directory for complete usage examples.

## 📁 Project Structure

```
WinstonAI/
├── winston_ai/                     # Main library package
│   ├── __init__.py                 # Package initialization
│   ├── models/                     # Neural network models
│   │   ├── winston_model.py        # WinstonAI & AdvancedWinstonAI models
│   │   └── attention.py            # Multi-head attention mechanism
│   ├── training/                   # Training utilities
│   │   ├── trainer.py              # High-level training orchestration
│   │   ├── agent.py                # DQN agent implementation
│   │   └── environment.py          # Trading environment simulation
│   ├── trading/                    # Live trading functionality
│   │   └── live_trader.py          # Live trading interface
│   ├── indicators/                 # Technical analysis
│   │   └── technical.py            # Technical indicators calculator
│   └── utils/                      # Utility functions
│       ├── config.py               # Configuration management
│       ├── device.py               # GPU/device management
│       └── checkpoints.py          # Model checkpoint utilities
├── examples/                       # Example scripts
│   ├── quickstart.py               # Quick start example
│   ├── train_model.py              # Full training example
│   ├── use_model.py                # Inference example
│   └── README.md                   # Examples documentation
├── src/                            # Legacy scripts (for reference)
│   ├── train_gpu_optimized.py      # GPU-optimized training script
│   ├── train_rl_5s.py              # RL trainer (5s timeframe)
│   ├── ultra_live_trading_bot.py   # High-performance trading bot
│   ├── live_trading_bot.py         # Standard live trading bot
│   └── gpu_monitor.py              # GPU monitoring utilities
├── data/                           # Data directory
│   └── configs/                    # Configuration files
│       ├── training_config.json    # Training configuration
│       ├── trading_config.json     # Trading configuration
│       └── gpu_config.json         # GPU settings
├── models/                         # Saved models (gitignored)
├── docs/                           # Documentation
├── tests/                          # Unit tests
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── README.md                       # This file
```

## 🎮 Usage Examples

### Quick Start

```bash
# Install the package
pip install -e .

# Run quick start example
python examples/quickstart.py
```

### Training a New Model (Library API)

```python
from winston_ai import Trainer, Config
import pandas as pd

# Load your market data
data = pd.read_csv('your_market_data.csv')
# Ensure data has columns: open, high, low, close, volume

# Configure training
config = Config()
config.update('training', 
    episodes=1000,
    batch_size=512,
    learning_rate=0.0001
)

# Create trainer and train
trainer = Trainer(data=data, config=config)
metrics = trainer.train(episodes=1000)

# Plot results
trainer.plot_results('training_results.png')
```

### Using a Trained Model

```python
from winston_ai import LiveTrader
import pandas as pd

# Load trained model
trader = LiveTrader(
    model_path='models/winston_ai_final.pth',
    lookback_window=100
)

# Get recent market data
data = get_recent_market_data()  # Your data source

# Make prediction
prediction = trader.predict(data)
print(f"Action: {prediction['action_name']}")
print(f"Confidence: {prediction['confidence']:.2%}")

# Check if should trade
if trader.should_trade(data, min_confidence=0.7):
    execute_trade(prediction['action_name'])
```

### Importing Models Directly

```python
from winston_ai import WinstonAI, AdvancedWinstonAI
import torch

# Create model
model = AdvancedWinstonAI(
    state_size=10000,
    action_size=3,  # HOLD, CALL, PUT
    hidden_size=4096
)

# Use for training or inference
model.eval()
with torch.no_grad():
    q_values = model(state_tensor)
    action = q_values.argmax().item()
```

For more examples, see the `examples/` directory.

## 📊 Model Architecture

WinstonAI uses a sophisticated deep learning architecture:

- **Input Layer:** 100+ features (technical indicators + market data)
- **Feature Extraction:** 4-layer LSTM (512 hidden units each)
- **Attention Mechanism:** Multi-head attention (16 heads, 256 dimensions)
- **Value & Advantage Streams:** Dueling DQN architecture
- **Output Layer:** 3 actions (CALL, PUT, HOLD)

### Training Features
- Experience replay buffer (1M transitions)
- Target network with soft updates
- Gradient clipping for stability
- Mixed precision training
- Dynamic learning rate scheduling

## 🔧 Configuration

### GPU Configuration (`gpu_config.json`)
```json
{
    "device": "cuda",
    "mixed_precision": true,
    "gradient_checkpointing": true,
    "memory_efficient": true
}
```

### Trading Configuration (`trading_config.json`)
```json
{
    "max_daily_loss": 100,
    "stop_loss_percent": 0.02,
    "take_profit_percent": 0.04,
    "trade_amount": 10
}
```

## 📈 Performance

- **Training Speed:** ~500-1000 episodes/hour (RTX 3060 Ti)
- **Inference Time:** <10ms per prediction
- **Memory Usage:** 8-10GB VRAM during training
- **Model Size:** ~500MB (full checkpoint)

## 🛠️ GPU Optimization Guide

See [README_GPU_OPTIMIZATION.md](src/README_GPU_OPTIMIZATION.md) for detailed information on:
- GPU configuration and setup
- Memory optimization techniques
- Performance tuning
- Troubleshooting common issues

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- Code of conduct
- Development setup
- Submitting pull requests
- Coding standards

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT:** This software is for educational and research purposes only. Trading binary options and forex carries significant financial risk. The developers are not responsible for any financial losses incurred while using this software. Always:

- Test thoroughly with demo accounts before live trading
- Never trade with money you cannot afford to lose
- Understand the risks of automated trading
- Comply with your local financial regulations
- Use proper risk management strategies

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Technical indicators powered by [ta-lib](https://github.com/mrjbq7/ta-lib) and [ta](https://github.com/bukosabino/ta)
- Trading API integration via [BinaryOptionsToolsV2](https://github.com/BinaryOptionsTools/BinaryOptionsToolsV2)
- Historical data from [Kaggle](https://www.kaggle.com/)

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/ChipaDevTeam/WinstonAI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ChipaDevTeam/WinstonAI/discussions)

## 🗺️ Roadmap

- [ ] Multi-GPU training support
- [ ] Enhanced ensemble learning
- [ ] Additional trading platforms integration
- [ ] Web dashboard for monitoring
- [ ] Backtesting framework improvements
- [ ] Paper trading mode
- [ ] Advanced strategy optimization

---

**Made with ❤️ by ChipaDevTeam**

*Star ⭐ this repository if you find it useful!*
