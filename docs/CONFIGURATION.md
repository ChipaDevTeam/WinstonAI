# Configuration Guide

This guide explains how to configure WinstonAI for training and trading.

## Configuration Files

WinstonAI uses JSON configuration files located in the `src/` directory:

- `training_config.json` - Training parameters
- `gpu_config.json` - GPU settings
- `trading_config.json` - Trading bot configuration
- `ultra_trading_config.json` - Ultra bot configuration

## Creating Local Configurations

To avoid committing sensitive data, create local configuration files:

```bash
cd src
cp training_config.json training_config.local.json
cp trading_config.json trading_config.local.json
```

Local files (`*.local.json`) are ignored by git.

## Training Configuration

### training_config.json

```json
{
  "episodes": 5000,
  "batch_size": 64,
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "epsilon_start": 1.0,
  "epsilon_end": 0.01,
  "epsilon_decay": 0.995,
  "memory_size": 1000000,
  "target_update": 10,
  "save_frequency": 100,
  "state_size": 100,
  "action_size": 3
}
```

**Parameters:**

- `episodes`: Number of training episodes
- `batch_size`: Number of samples per training batch
- `learning_rate`: Learning rate for optimizer
- `gamma`: Discount factor for future rewards
- `epsilon_start`: Initial exploration rate
- `epsilon_end`: Final exploration rate
- `epsilon_decay`: Decay rate for exploration
- `memory_size`: Replay buffer size
- `target_update`: Episodes between target network updates
- `save_frequency`: Save model every N episodes
- `state_size`: Number of input features
- `action_size`: Number of possible actions (CALL, PUT, HOLD)

### Tuning Training Parameters

**For faster training (less accuracy):**
```json
{
  "episodes": 1000,
  "batch_size": 128,
  "learning_rate": 0.001
}
```

**For better accuracy (slower):**
```json
{
  "episodes": 10000,
  "batch_size": 32,
  "learning_rate": 0.00001
}
```

## GPU Configuration

### gpu_config.json

```json
{
  "device": "cuda",
  "mixed_precision": true,
  "gradient_checkpointing": true,
  "memory_efficient": true,
  "batch_size_multiplier": 1,
  "num_workers": 4
}
```

**Parameters:**

- `device`: "cuda" for GPU, "cpu" for CPU
- `mixed_precision`: Use FP16/FP32 mixed precision
- `gradient_checkpointing`: Save memory during training
- `memory_efficient`: Enable memory optimizations
- `batch_size_multiplier`: Multiply batch size by this factor
- `num_workers`: Data loading threads

### GPU Memory Optimization

**For 8GB VRAM:**
```json
{
  "mixed_precision": true,
  "gradient_checkpointing": true,
  "batch_size_multiplier": 0.5
}
```

**For 12GB+ VRAM:**
```json
{
  "mixed_precision": true,
  "gradient_checkpointing": false,
  "batch_size_multiplier": 2
}
```

## Trading Configuration

### trading_config.json

```json
{
  "email": "your_email@example.com",
  "password": "your_password",
  "assets": ["EURUSD", "GBPUSD", "USDJPY"],
  "timeframe": 60,
  "trade_amount": 10,
  "max_daily_loss": 100,
  "stop_loss_percent": 0.02,
  "take_profit_percent": 0.04,
  "max_concurrent_trades": 3,
  "risk_per_trade": 0.02,
  "min_confidence": 0.7
}
```

**Parameters:**

- `email`: PocketOption email
- `password`: PocketOption password (or use environment variable)
- `assets`: Trading pairs to monitor
- `timeframe`: Candle timeframe in seconds
- `trade_amount`: Amount per trade (in account currency)
- `max_daily_loss`: Stop trading after this loss
- `stop_loss_percent`: Stop loss as % of trade amount
- `take_profit_percent`: Take profit as % of trade amount
- `max_concurrent_trades`: Maximum simultaneous trades
- `risk_per_trade`: Risk per trade as % of account
- `min_confidence`: Minimum confidence to execute trade

### Risk Management Settings

**Conservative (Low Risk):**
```json
{
  "trade_amount": 5,
  "max_daily_loss": 50,
  "stop_loss_percent": 0.01,
  "max_concurrent_trades": 1,
  "risk_per_trade": 0.01,
  "min_confidence": 0.8
}
```

**Aggressive (High Risk):**
```json
{
  "trade_amount": 20,
  "max_daily_loss": 200,
  "stop_loss_percent": 0.05,
  "max_concurrent_trades": 5,
  "risk_per_trade": 0.05,
  "min_confidence": 0.6
}
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# PocketOption Credentials
POCKETOPTION_EMAIL=your_email@example.com
POCKETOPTION_PASSWORD=your_password

# API Keys (if needed)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Logging
LOG_LEVEL=INFO
LOG_FILE=winston_ai.log

# GPU
CUDA_VISIBLE_DEVICES=0
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
email = os.getenv('POCKETOPTION_EMAIL')
```

## Asset Configuration

### Supported Assets

Common forex pairs:
- EURUSD - Euro / US Dollar
- GBPUSD - British Pound / US Dollar
- USDJPY - US Dollar / Japanese Yen
- AUDUSD - Australian Dollar / US Dollar
- USDCAD - US Dollar / Canadian Dollar
- NZDUSD - New Zealand Dollar / US Dollar
- EURJPY - Euro / Japanese Yen
- GBPJPY - British Pound / Japanese Yen

### Asset Selection

Choose assets based on:
- **Liquidity:** Higher is better
- **Volatility:** Match to your strategy
- **Trading hours:** Ensure overlap with your availability
- **Spread:** Lower is better

## Timeframe Configuration

Available timeframes (in seconds):

- `60` - 1 minute
- `300` - 5 minutes
- `900` - 15 minutes
- `1800` - 30 minutes
- `3600` - 1 hour
- `14400` - 4 hours
- `86400` - 1 day

**Recommended:**
- Beginners: 300-900 (5-15 minutes)
- Intermediate: 60-300 (1-5 minutes)
- Advanced: 5-60 (5 seconds - 1 minute)

## Model Configuration

In Python code:

```python
from train_gpu_optimized import AdvancedWinstonAI

model = AdvancedWinstonAI(
    state_size=100,      # Number of input features
    action_size=3,       # CALL, PUT, HOLD
    hidden_size=512,     # LSTM hidden units
    num_layers=4,        # LSTM layers
    num_heads=16,        # Attention heads
    dropout=0.2,         # Dropout rate
    device='cuda'        # 'cuda' or 'cpu'
)
```

## Validation

Validate your configuration:

```python
import json

# Load config
with open('training_config.json', 'r') as f:
    config = json.load(f)

# Validate
assert config['batch_size'] > 0
assert 0 < config['learning_rate'] < 1
assert config['action_size'] == 3  # CALL, PUT, HOLD
```

## Best Practices

1. **Version Control:**
   - Never commit API keys or passwords
   - Use `*.local.json` for sensitive data
   - Use environment variables for secrets

2. **Testing:**
   - Test configuration with demo account first
   - Start with small trade amounts
   - Gradually increase as confidence grows

3. **Monitoring:**
   - Monitor logs regularly
   - Set up alerts for errors
   - Review performance daily

4. **Backup:**
   - Backup configuration files
   - Keep multiple versions
   - Document changes

## Common Configuration Issues

**Issue:** Out of memory during training
**Solution:** Reduce batch_size or enable gradient_checkpointing

**Issue:** Trading bot not executing trades
**Solution:** Check min_confidence threshold and model confidence scores

**Issue:** Poor training performance
**Solution:** Adjust learning_rate, increase episodes, or tune hyperparameters

**Issue:** API connection errors
**Solution:** Verify credentials, check internet connection, ensure API is accessible

## Configuration Templates

### Development
```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  "trade_amount": 1
}
```

### Production
```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  "trade_amount": 10
}
```

## See Also

- [Installation Guide](INSTALLATION.md)
- [GPU Optimization Guide](../src/README_GPU_OPTIMIZATION.md)
- [Usage Examples](../README.md#usage-examples)

---

For more help, see the [GitHub Issues](https://github.com/ChipaDevTeam/WinstonAI/issues) or discussions.
