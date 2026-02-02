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

4. Configure your settings:
```bash
# Edit the configuration files in src/
cp src/training_config.json src/training_config.local.json
cp src/trading_config.json src/trading_config.local.json
# Update with your API keys and preferences
```

### Training the Model

**Quick Start (GPU):**
```bash
cd src
python quick_start_gpu.py
```

**Full Training:**
```bash
cd src
python train_gpu_optimized.py
```

**Reinforcement Learning Training (5-second timeframe):**
```bash
cd src
python train_rl_5s.py
```

### Live Trading

⚠️ **Warning:** Live trading involves real financial risk. Always test thoroughly with a demo account first.

```bash
cd src
python ultra_live_trading_bot.py
```

## 📁 Project Structure

```
WinstonAI/
├── src/
│   ├── train_gpu_optimized.py      # GPU-optimized training script
│   ├── train_rl_5s.py              # Reinforcement learning trainer (5s)
│   ├── ultra_live_trading_bot.py   # High-performance live trading bot
│   ├── live_trading_bot.py         # Standard live trading bot
│   ├── gpu_monitor.py              # GPU monitoring utilities
│   ├── gpu_benchmark.py            # GPU performance benchmarking
│   ├── quick_start_gpu.py          # Quick start script for GPU training
│   ├── download.py                 # Historical data downloader
│   ├── gethistory.py               # Historical data fetcher
│   ├── training_config.json        # Training configuration
│   ├── gpu_config.json             # GPU settings
│   ├── trading_config.json         # Trading bot configuration
│   ├── ultra_trading_config.json   # Ultra bot configuration
│   └── README_GPU_OPTIMIZATION.md  # Detailed GPU optimization guide
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── LICENSE                         # MIT License
├── CONTRIBUTING.md                 # Contribution guidelines
├── CODE_OF_CONDUCT.md              # Code of conduct
├── CHANGELOG.md                    # Version history
└── README.md                       # This file
```

## 🎮 Usage Examples

### Training a New Model

```python
from train_gpu_optimized import AdvancedWinstonAI

# Create model
model = AdvancedWinstonAI(
    state_size=100,
    action_size=3,  # CALL, PUT, HOLD
    device='cuda'
)

# Train
model.train(episodes=5000)
```

### Making Predictions

```python
import torch
from train_gpu_optimized import AdvancedWinstonAI

# Load trained model
model = torch.load('winston_ai_final.pth')
model.eval()

# Prepare state (your market data)
state = prepare_market_data()  # Your function to get market data

# Get action
with torch.no_grad():
    action = model.act(state)
```

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
