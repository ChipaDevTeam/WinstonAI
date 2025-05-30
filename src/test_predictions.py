# WinstonAI Quick Test Script
# Simple script to test the model prediction without live trading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ta
from sklearn.preprocessing import MinMaxScaler
import warnings
import asyncio

# Import the BinaryOptionsToolsV2 library
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

warnings.filterwarnings('ignore')

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import classes from the training script (we'll need the exact same structure)
class TechnicalIndicators:
    @staticmethod
    def calculate_indicators(df):
        df = df.copy()
        
        # Moving Averages
        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = ta.trend.macd_diff(df['close'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Bollinger Bands
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=20)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # CCI
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Price momentum
        df['price_momentum'] = df['close'].rolling(window=14).mean() / df['close'].rolling(window=28).mean()
        
        # Price Action Features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # Fibonacci Levels
        df['recent_high'] = df['high'].rolling(window=50).max()
        df['recent_low'] = df['low'].rolling(window=50).min()
        fib_range = df['recent_high'] - df['recent_low']
        df['fib_23.6'] = df['recent_high'] - 0.236 * fib_range
        df['fib_38.2'] = df['recent_high'] - 0.382 * fib_range
        df['fib_50.0'] = df['recent_high'] - 0.500 * fib_range
        df['fib_61.8'] = df['recent_high'] - 0.618 * fib_range
        
        # Support and Resistance
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        df['distance_to_support'] = (df['close'] - df['support']) / df['support']
        df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        
        # Chart Patterns
        df['doji'] = TechnicalIndicators.detect_doji(df)
        df['hammer'] = TechnicalIndicators.detect_hammer(df)
        df['shooting_star'] = TechnicalIndicators.detect_shooting_star(df)
        df['engulfing_bullish'] = TechnicalIndicators.detect_bullish_engulfing(df)
        df['engulfing_bearish'] = TechnicalIndicators.detect_bearish_engulfing(df)
        
        return df
    
    @staticmethod
    def detect_doji(df):
        body_size = abs(df['close'] - df['open'])
        candle_range = df['high'] - df['low']
        return (body_size / candle_range) < 0.1
    
    @staticmethod
    def detect_hammer(df):
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        return (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
    
    @staticmethod
    def detect_shooting_star(df):
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        return (upper_shadow > 2 * body_size) & (lower_shadow < body_size)
    
    @staticmethod
    def detect_bullish_engulfing(df):
        prev_bearish = df['close'].shift(1) < df['open'].shift(1)
        curr_bullish = df['close'] > df['open']
        engulfing = (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
        return prev_bearish & curr_bullish & engulfing
    
    @staticmethod
    def detect_bearish_engulfing(df):
        prev_bullish = df['close'].shift(1) > df['open'].shift(1)
        curr_bearish = df['close'] < df['open']
        engulfing = (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
        return prev_bullish & curr_bearish & engulfing

class WinstonAI(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(WinstonAI, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size // 4,
            hidden_size=hidden_size // 8,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.decision_layers = nn.Sequential(
            nn.Linear(hidden_size // 8, hidden_size // 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 16, action_size)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.unsqueeze(1)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]
        q_values = self.decision_layers(lstm_out)
        return q_values

class WinstonAITester:
    def __init__(self, model_path):
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback_window = 50
        self.load_model(model_path)
    def load_model(self, model_path):
        # Calculate state size using sufficient sample data for indicators
        sample_data = pd.DataFrame({
            'open': [1.0] * 100,
            'high': [1.1] * 100,
            'close': [1.0] * 100,
            'low': [0.9] * 100
        })
        sample_indicators = TechnicalIndicators.calculate_indicators(sample_data)
        feature_columns = [col for col in sample_indicators.columns if col not in ['time', 'asset']]
        state_size = len(feature_columns) * self.lookback_window        
        # Load model
        try:
            self.model = WinstonAI(state_size=state_size, action_size=3).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['q_network'])
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 State size: {state_size}")
            print(f"📁 Model path: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    def prepare_state(self, candles_df):
        if len(candles_df) < self.lookback_window:
            print(f"⚠️  Not enough data: {len(candles_df)} < {self.lookback_window}")
            return None
        
        try:
            df_with_indicators = TechnicalIndicators.calculate_indicators(candles_df)
            feature_columns = [col for col in df_with_indicators.columns if col not in ['time', 'asset']]
            
            # Handle NaN values
            df_with_indicators = df_with_indicators.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Check if we have enough valid data after indicator calculation
            if len(df_with_indicators.dropna()) < self.lookback_window:
                print(f"⚠️  Not enough valid data after indicators calculation")
                return None
            
            scaled_data = self.scaler.fit_transform(df_with_indicators[feature_columns])
            state = scaled_data[-self.lookback_window:].flatten()
            
            return state
        except Exception as e:
            print(f"❌ Error preparing state: {e}")
            return None
    
    def predict_action(self, state):
        if state is None:
            return 0, 0.0
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            confidence = torch.softmax(q_values, dim=1).max().cpu().numpy()
            
            return action, confidence

async def test_predictions(ssid):
    """Test the model predictions with live data"""
    print("🤖 Testing WinstonAI Predictions")
    print("=" * 40)
      # Initialize tester
    tester = WinstonAITester(r"c:\Users\tp\ComunityPrograms\WinstonAI\winston_ai_episode_400.pth")
    
    # Initialize API
    api = PocketOptionAsync(ssid)
    await asyncio.sleep(5)
    
    # Get balance
    balance = await api.balance()
    print(f"💰 Account Balance: ${balance:.2f}")
    
    # Test assets
    assets = ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"]
    
    for asset in assets:
        print(f"\n🔍 Testing {asset}:")
        
        try:
            # Get recent candles
            candles = await api.get_candles(asset, 60, 3600)
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])
            
            # Prepare state
            state = tester.prepare_state(df)
            
            if state is not None:
                # Get prediction
                action, confidence = tester.predict_action(state)
                
                action_names = ["HOLD", "CALL", "PUT"]
                print(f"  🤖 Prediction: {action_names[action]}")
                print(f"  📊 Confidence: {confidence:.3f}")
                print(f"  💹 Current Price: {df.iloc[-1]['close']:.5f}")
                
                # Get payout for this asset
                payout = await api.payout(asset)
                print(f"  💰 Payout: {payout:.1f}%")
                
                if confidence > 0.7:
                    print(f"  ✅ HIGH CONFIDENCE - Trade recommended!")
                else:
                    print(f"  ⚠️  Low confidence - Consider waiting")
            else:
                print(f"  ❌ Not enough data for prediction")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 40)
    print("Test completed! Use live_trading_bot.py for actual trading.")

if __name__ == "__main__":
    ssid = input("Enter your PocketOption SSID: ").strip()
    asyncio.run(test_predictions(ssid))
