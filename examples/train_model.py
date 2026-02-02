"""
Example: Training a WinstonAI model

This script demonstrates how to train a WinstonAI model using the library API.
"""

import pandas as pd
import numpy as np
from winston_ai import Trainer, Config

def generate_sample_data(n_samples=10000):
    """
    Generate sample OHLCV data for demonstration
    
    In production, replace this with real market data from your data source.
    """
    print("[Data] Generating sample data...")
    
    # Create synthetic price data with random walk
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n_samples) * 0.1,
        'high': close_prices + np.abs(np.random.randn(n_samples) * 0.2),
        'low': close_prices - np.abs(np.random.randn(n_samples) * 0.2),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= close >= low
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    
    print(f"[Data] Generated {len(data)} samples")
    return data


def main():
    """Main training function"""
    
    print("=" * 70)
    print("WinstonAI Training Example")
    print("=" * 70)
    
    # Step 1: Load or generate data
    # In production, load real market data:
    # data = pd.read_csv('your_market_data.csv')
    data = generate_sample_data(n_samples=10000)
    
    # Step 2: Configure training parameters
    config = Config()
    config.update('training',
        episodes=500,              # Number of training episodes
        batch_size=512,            # Batch size for training
        learning_rate=0.0001,      # Learning rate
        gamma=0.99,                # Discount factor
        epsilon_start=1.0,         # Initial exploration rate
        epsilon_end=0.01,          # Final exploration rate
        epsilon_decay=0.995,       # Exploration decay rate
        save_frequency=50          # Save checkpoint every N episodes
    )
    
    config.update('trading',
        payout_ratio=0.8,          # 80% profit on winning trades
        trade_amount=100,          # Amount per trade
        initial_balance=10000      # Starting balance
    )
    
    print("\n[Config] Training configuration:")
    print(f"  Episodes: {config.training['episodes']}")
    print(f"  Batch size: {config.training['batch_size']}")
    print(f"  Learning rate: {config.training['learning_rate']}")
    
    # Step 3: Create trainer
    trainer = Trainer(
        data=data,
        config=config,
        checkpoint_dir="models"
    )
    
    # Step 4: Train the model
    print("\n[Training] Starting training...")
    metrics = trainer.train(
        episodes=config.training['episodes'],
        save_frequency=config.training['save_frequency'],
        verbose=True
    )
    
    # Step 5: Plot and save results
    print("\n[Results] Plotting training results...")
    trainer.plot_results(save_path="winston_ai_training_results.png")
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Total Episodes: {len(metrics['rewards'])}")
    print(f"Average Reward (last 100): {np.mean(metrics['rewards'][-100:]):.2f}")
    print(f"Average Win Rate (last 100): {np.mean(metrics['win_rates'][-100:]):.2%}")
    print(f"Final Balance (last 100): ${np.mean(metrics['balances'][-100:]):.2f}")
    print(f"Best Win Rate: {max(metrics['win_rates']):.2%}")
    print(f"Best Balance: ${max(metrics['balances']):.2f}")
    print("=" * 70)
    
    print("\n[Complete] Training finished!")
    print(f"[Model] Saved to: models/winston_ai_final.pth")
    print(f"[Plot] Saved to: winston_ai_training_results.png")


if __name__ == "__main__":
    main()
