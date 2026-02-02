"""
Example: Using a trained WinstonAI model for predictions

This script demonstrates how to load a trained model and use it for making predictions.
"""

import pandas as pd
import numpy as np
import torch
from winston_ai.trading import LiveTrader


def generate_sample_data(n_samples=200):
    """
    Generate sample market data
    
    In production, this would be real-time market data from your broker API.
    """
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n_samples) * 0.1,
        'high': close_prices + np.abs(np.random.randn(n_samples) * 0.2),
        'low': close_prices - np.abs(np.random.randn(n_samples) * 0.2),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    
    return data


def main():
    """Main prediction function"""
    
    print("=" * 70)
    print("WinstonAI Inference Example")
    print("=" * 70)
    
    # Step 1: Load the trained model
    model_path = "models/winston_ai_final.pth"
    
    print(f"\n[Model] Loading model from {model_path}")
    
    try:
        trader = LiveTrader(
            model_path=model_path,
            lookback_window=100,
            use_advanced_model=True
        )
    except FileNotFoundError:
        print(f"\n[Error] Model not found at {model_path}")
        print("[Info] Please train a model first using examples/train_model.py")
        return
    
    # Step 2: Get market data
    # In production, this would come from your broker's API
    print("\n[Data] Loading market data...")
    data = generate_sample_data(n_samples=200)
    print(f"[Data] Loaded {len(data)} candles")
    
    # Step 3: Make predictions
    print("\n[Prediction] Making prediction...")
    prediction = trader.predict(data)
    
    # Display results
    print("\n" + "=" * 70)
    print("Prediction Results")
    print("=" * 70)
    print(f"Action: {prediction['action_name']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"\nQ-Values:")
    print(f"  HOLD: {prediction['q_values'][0]:.4f}")
    print(f"  CALL: {prediction['q_values'][1]:.4f}")
    print(f"  PUT:  {prediction['q_values'][2]:.4f}")
    print("=" * 70)
    
    # Step 4: Trading decision
    min_confidence = 0.6
    print(f"\n[Decision] Checking if should trade (min confidence: {min_confidence:.0%})")
    
    should_trade = trader.should_trade(data, min_confidence=min_confidence)
    
    if should_trade:
        print(f"✓ EXECUTE {prediction['action_name']} trade")
        print(f"  Reason: Confidence {prediction['confidence']:.2%} >= {min_confidence:.0%}")
    else:
        if prediction['action_name'] == 'HOLD':
            print("○ HOLD position")
            print(f"  Reason: Model recommends holding")
        else:
            print(f"✗ DO NOT trade")
            print(f"  Reason: Confidence {prediction['confidence']:.2%} < {min_confidence:.0%}")
    
    # Step 5: Simulate multiple predictions
    print("\n[Simulation] Testing on multiple time windows...")
    print("-" * 70)
    
    predictions_summary = {
        'HOLD': 0,
        'CALL': 0,
        'PUT': 0
    }
    
    # Test on sliding windows
    window_size = 100
    step_size = 10
    
    for i in range(window_size, len(data) - 10, step_size):
        window_data = data.iloc[i-window_size:i].copy()
        pred = trader.predict(window_data)
        predictions_summary[pred['action_name']] += 1
    
    total_predictions = sum(predictions_summary.values())
    
    print("\nPrediction Distribution:")
    for action, count in predictions_summary.items():
        percentage = (count / total_predictions) * 100
        print(f"  {action}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n[Complete] Inference example finished!")


if __name__ == "__main__":
    main()
