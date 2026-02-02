# API Documentation

This document describes the main APIs and interfaces in WinstonAI.

## Model API

### AdvancedWinstonAI

The main model class for trading decisions.

```python
from train_gpu_optimized import AdvancedWinstonAI

model = AdvancedWinstonAI(
    state_size: int = 100,      # Number of input features
    action_size: int = 3,       # Number of actions (CALL, PUT, HOLD)
    hidden_size: int = 512,     # LSTM hidden units
    num_layers: int = 4,        # Number of LSTM layers
    num_heads: int = 16,        # Multi-head attention heads
    dropout: float = 0.2,       # Dropout rate
    device: str = 'cuda'        # Device ('cuda' or 'cpu')
)
```

#### Methods

##### `act(state, epsilon=0.0)`
Get action for a given state.

**Parameters:**
- `state` (np.ndarray): Current state observation
- `epsilon` (float): Exploration rate (default: 0.0)

**Returns:**
- `int`: Action index (0=CALL, 1=PUT, 2=HOLD)

**Example:**
```python
state = get_current_state()  # Your function
action = model.act(state, epsilon=0.1)
```

##### `remember(state, action, reward, next_state, done)`
Store experience in replay buffer.

**Parameters:**
- `state` (np.ndarray): Current state
- `action` (int): Action taken
- `reward` (float): Reward received
- `next_state` (np.ndarray): Next state
- `done` (bool): Whether episode ended

**Example:**
```python
model.remember(state, action, reward, next_state, done)
```

##### `replay(batch_size=64)`
Train on a batch from replay buffer.

**Parameters:**
- `batch_size` (int): Number of samples to train on

**Returns:**
- `float`: Training loss

**Example:**
```python
loss = model.replay(batch_size=128)
```

##### `load(filepath)`
Load model from file.

**Parameters:**
- `filepath` (str): Path to model file

**Example:**
```python
model.load('winston_ai_final.pth')
```

##### `save(filepath)`
Save model to file.

**Parameters:**
- `filepath` (str): Path to save model

**Example:**
```python
model.save('my_model.pth')
```

## Trading Bot API

### UltraLiveTradingBot

High-performance live trading bot.

```python
from ultra_live_trading_bot import UltraLiveTradingBot

bot = UltraLiveTradingBot(
    model_path: str,           # Path to trained model
    config: dict               # Trading configuration
)
```

#### Methods

##### `async start()`
Start the trading bot.

**Example:**
```python
import asyncio

bot = UltraLiveTradingBot(
    model_path='winston_ai_final.pth',
    config=config
)

asyncio.run(bot.start())
```

##### `async stop()`
Stop the trading bot gracefully.

**Example:**
```python
await bot.stop()
```

## Technical Indicators API

### AdvancedTechnicalIndicators

Calculate technical indicators.

```python
from train_gpu_optimized import AdvancedTechnicalIndicators

indicators = AdvancedTechnicalIndicators()
```

#### Methods

##### `calculate_all_indicators(df)`
Calculate all technical indicators for a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): OHLCV data with columns: open, high, low, close, volume

**Returns:**
- `pd.DataFrame`: DataFrame with all indicators added

**Example:**
```python
df = pd.read_csv('price_data.csv')
df_with_indicators = indicators.calculate_all_indicators(df)
```

##### `calculate_live_indicators(df)`
Calculate indicators optimized for live trading.

**Parameters:**
- `df` (pd.DataFrame): Recent OHLCV data (last N candles)

**Returns:**
- `dict`: Dictionary of indicator values

**Example:**
```python
recent_data = get_recent_candles(limit=100)
indicators = indicators.calculate_live_indicators(recent_data)
```

## Data Processing API

### Data Loading

```python
import pandas as pd

# Load historical data
df = pd.read_csv('historical_data.csv')

# Expected columns
required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
```

### Data Preprocessing

```python
from sklearn.preprocessing import MinMaxScaler

# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
```

## Configuration API

### Loading Configuration

```python
import json

# Load training config
with open('training_config.json', 'r') as f:
    config = json.load(f)

# Load trading config
with open('trading_config.json', 'r') as f:
    trading_config = json.load(f)
```

### Configuration Structure

#### Training Configuration
```python
{
    "episodes": int,              # Number of training episodes
    "batch_size": int,            # Batch size for training
    "learning_rate": float,       # Learning rate
    "gamma": float,               # Discount factor
    "epsilon_start": float,       # Initial exploration rate
    "epsilon_end": float,         # Final exploration rate
    "epsilon_decay": float,       # Exploration decay rate
    "memory_size": int,           # Replay buffer size
    "target_update": int,         # Target network update frequency
    "save_frequency": int         # Model save frequency
}
```

#### Trading Configuration
```python
{
    "email": str,                 # Trading account email
    "password": str,              # Trading account password
    "assets": List[str],          # Assets to trade
    "timeframe": int,             # Candle timeframe (seconds)
    "trade_amount": float,        # Amount per trade
    "max_daily_loss": float,      # Maximum daily loss
    "stop_loss_percent": float,   # Stop loss percentage
    "take_profit_percent": float, # Take profit percentage
    "max_concurrent_trades": int, # Max simultaneous trades
    "min_confidence": float       # Minimum confidence threshold
}
```

## GPU Utilities API

### GPU Monitoring

```python
from gpu_monitor import GPUMonitor

monitor = GPUMonitor()
stats = monitor.get_stats()

print(f"GPU Memory: {stats['memory_used']}/{stats['memory_total']} MB")
print(f"GPU Utilization: {stats['gpu_util']}%")
```

### GPU Benchmarking

```python
from gpu_benchmark import run_benchmark

results = run_benchmark()
print(f"Operations/second: {results['ops_per_second']}")
print(f"Memory bandwidth: {results['memory_bandwidth']} GB/s")
```

## Error Handling

All methods raise standard Python exceptions:

```python
try:
    model.load('nonexistent_model.pth')
except FileNotFoundError:
    print("Model file not found")
except Exception as e:
    print(f"Error loading model: {e}")
```

## Common Patterns

### Training Loop

```python
import torch
from train_gpu_optimized import AdvancedWinstonAI

# Create model
model = AdvancedWinstonAI(state_size=100, action_size=3)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Choose action
        action = model.act(state, epsilon=epsilon)
        
        # Take action
        next_state, reward, done = env.step(action)
        
        # Store experience
        model.remember(state, action, reward, next_state, done)
        
        # Train
        if len(model.memory) > batch_size:
            loss = model.replay(batch_size)
        
        state = next_state
        total_reward += reward
    
    # Save periodically
    if episode % 100 == 0:
        model.save(f'model_episode_{episode}.pth')
```

### Live Trading Loop

```python
import asyncio
from ultra_live_trading_bot import UltraLiveTradingBot

async def main():
    # Initialize bot
    bot = UltraLiveTradingBot(
        model_path='winston_ai_final.pth',
        config=trading_config
    )
    
    # Start trading
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()

if __name__ == '__main__':
    asyncio.run(main())
```

## Type Hints

WinstonAI uses type hints for better code clarity:

```python
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

def process_state(
    data: np.ndarray,
    indicators: Dict[str, float]
) -> torch.Tensor:
    """Process state with type hints."""
    pass
```

## Constants

Common constants used throughout the codebase:

```python
# Action indices
ACTION_CALL = 0
ACTION_PUT = 1
ACTION_HOLD = 2

# Devices
DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

# Timeframes (seconds)
TIMEFRAME_1M = 60
TIMEFRAME_5M = 300
TIMEFRAME_15M = 900
TIMEFRAME_1H = 3600
```

## See Also

- [Configuration Guide](CONFIGURATION.md)
- [Installation Guide](INSTALLATION.md)
- [README](../README.md)

---

For more examples, see the source code in `src/` directory.
