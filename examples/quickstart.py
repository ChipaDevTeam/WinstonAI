"""
Example: Simple library usage demonstration

This script shows the most basic way to use WinstonAI library.
"""

import pandas as pd
import numpy as np

# Import WinstonAI components
from winston_ai import Trainer, Config, WinstonAI, AdvancedWinstonAI


def create_sample_data():
    """Create sample OHLCV data"""
    np.random.seed(42)
    n = 5000
    
    # Generate random walk price data
    prices = 100 + np.cumsum(np.random.randn(n) * 0.02)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(n) * 0.1,
        'high': prices + np.abs(np.random.randn(n) * 0.2),
        'low': prices - np.abs(np.random.randn(n) * 0.2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    
    return data


def main():
    print("WinstonAI - Quick Start Example")
    print("=" * 50)
    
    # 1. Create or load data
    print("\n1. Creating sample data...")
    data = create_sample_data()
    print(f"   Created {len(data)} data points")
    
    # 2. Configure training
    print("\n2. Configuring training...")
    config = Config()
    config.update('training', episodes=100, batch_size=256)
    print(f"   Episodes: {config.training['episodes']}")
    print(f"   Batch size: {config.training['batch_size']}")
    
    # 3. Create trainer
    print("\n3. Creating trainer...")
    trainer = Trainer(data=data, config=config, checkpoint_dir="models")
    print(f"   State size: {trainer.env.state_size}")
    print(f"   Action size: {trainer.env.action_space}")
    
    # 4. Train model
    print("\n4. Training model...")
    print("   (This may take a few minutes)")
    metrics = trainer.train(
        episodes=100,
        save_frequency=25,
        verbose=False  # Set to True for detailed output
    )
    
    # 5. Show results
    print("\n5. Training results:")
    print(f"   Final reward: {metrics['rewards'][-1]:.2f}")
    print(f"   Final win rate: {metrics['win_rates'][-1]:.2%}")
    print(f"   Final balance: ${metrics['balances'][-1]:.2f}")
    
    # 6. Save plot
    print("\n6. Saving results plot...")
    trainer.plot_results(save_path="quickstart_results.png")
    print("   Plot saved to: quickstart_results.png")
    
    print("\n" + "=" * 50)
    print("Quick start complete!")
    print("\nNext steps:")
    print("  - View the training plot: quickstart_results.png")
    print("  - Use the trained model: examples/use_model.py")
    print("  - Train with more episodes for better results")
    

if __name__ == "__main__":
    main()
