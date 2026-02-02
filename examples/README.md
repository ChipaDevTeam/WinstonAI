# WinstonAI Examples

This directory contains example scripts demonstrating how to use the WinstonAI library.

## Examples

### 1. Quick Start (`quickstart.py`)
The simplest example to get started with WinstonAI.

```bash
python examples/quickstart.py
```

This script:
- Creates sample data
- Configures and trains a model for 100 episodes
- Saves results and plots

**Time:** ~5-10 minutes (depending on hardware)

### 2. Training a Model (`train_model.py`)
Comprehensive example of training a WinstonAI model with full configuration.

```bash
python examples/train_model.py
```

This script:
- Demonstrates full configuration options
- Trains for 500 episodes (configurable)
- Saves checkpoints during training
- Generates detailed training plots

**Time:** ~30-60 minutes (depending on hardware)

### 3. Using a Trained Model (`use_model.py`)
Example of loading and using a trained model for predictions.

```bash
python examples/use_model.py
```

This script:
- Loads a trained model
- Makes predictions on market data
- Shows trading decisions with confidence scores
- Demonstrates prediction on multiple time windows

**Requirements:** You must train a model first using one of the training examples.

## Modifying the Examples

### Using Your Own Data

Replace the `generate_sample_data()` function with your own data loader:

```python
# Instead of:
data = generate_sample_data()

# Use:
data = pd.read_csv('your_market_data.csv')
# Ensure columns: open, high, low, close, volume
```

### Adjusting Training Parameters

Edit the configuration in the training scripts:

```python
config.update('training',
    episodes=1000,           # More episodes = better learning
    batch_size=512,          # Larger = faster but more memory
    learning_rate=0.0001,    # Lower = more stable
    epsilon_decay=0.995      # Slower decay = more exploration
)
```

### Changing Model Architecture

Switch between standard and advanced models:

```python
# In trainer initialization:
trainer = Trainer(
    data=data,
    config=config,
    use_advanced_model=False  # Use standard WinstonAI model
)
```

## GPU vs CPU

The examples automatically detect and use GPU if available. To force CPU:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Add at top of script
```

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in configuration
- Use standard model instead of advanced
- Close other applications

### Slow Training
- Ensure GPU is being used (check console output)
- Reduce model size or number of episodes
- Use smaller dataset

### Poor Performance
- Train for more episodes (2000+)
- Adjust exploration parameters
- Use more diverse training data
- Tune learning rate and other hyperparameters

## Next Steps

After running these examples:

1. Train with real market data
2. Integrate with your broker's API for live trading
3. Implement proper backtesting
4. Add risk management strategies
5. Optimize hyperparameters for your specific market

## Note

⚠️ **Warning:** These examples use synthetic data for demonstration. Real trading requires:
- Real historical market data
- Proper backtesting and validation
- Risk management systems
- Live API integration
- Regulatory compliance

Never trade real money without thorough testing!
