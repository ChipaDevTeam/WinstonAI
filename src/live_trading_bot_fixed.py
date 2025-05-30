# WinstonAI Live Trading Bot - Fixed Version
# Real-time binary options trading using trained WinstonAI model with PocketOption

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import asyncio
import ta
from sklearn.preprocessing import MinMaxScaler
import warnings
import json
import time
from datetime import datetime, timedelta
import logging

# Import the BinaryOptionsToolsV2 library
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from BinaryOptionsToolsV2.tracing import start_logs

warnings.filterwarnings('ignore')

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TechnicalIndicators:
    """
    Comprehensive technical analysis indicators for binary options trading
    """
    
    @staticmethod
    def calculate_indicators(df):
        """Calculate all technical indicators"""
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
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # CCI (Commodity Channel Index)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Price-based momentum indicator
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
        
        # Support and Resistance levels
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        df['distance_to_support'] = (df['close'] - df['support']) / df['support']
        df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        
        # Chart Patterns Detection
        df['doji'] = TechnicalIndicators.detect_doji(df)
        df['hammer'] = TechnicalIndicators.detect_hammer(df)
        df['shooting_star'] = TechnicalIndicators.detect_shooting_star(df)
        df['engulfing_bullish'] = TechnicalIndicators.detect_bullish_engulfing(df)
        df['engulfing_bearish'] = TechnicalIndicators.detect_bearish_engulfing(df)
        
        # Fill NaN values to ensure clean data
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    @staticmethod
    def detect_doji(df):
        """Detect Doji candlestick pattern"""
        body_size = abs(df['close'] - df['open'])
        candle_range = df['high'] - df['low']
        return (body_size / candle_range) < 0.1
    
    @staticmethod
    def detect_hammer(df):
        """Detect Hammer candlestick pattern"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        return (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
    
    @staticmethod
    def detect_shooting_star(df):
        """Detect Shooting Star candlestick pattern"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        return (upper_shadow > 2 * body_size) & (lower_shadow < body_size)
    
    @staticmethod
    def detect_bullish_engulfing(df):
        """Detect Bullish Engulfing pattern"""
        prev_bearish = df['close'].shift(1) < df['open'].shift(1)
        curr_bullish = df['close'] > df['open']
        engulfing = (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
        return prev_bearish & curr_bullish & engulfing
    
    @staticmethod
    def detect_bearish_engulfing(df):
        """Detect Bearish Engulfing pattern"""
        prev_bullish = df['close'].shift(1) > df['open'].shift(1)
        curr_bearish = df['close'] < df['open']
        engulfing = (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
        return prev_bullish & curr_bearish & engulfing

class WinstonAI(nn.Module):
    """
    Deep Q-Network for binary options trading (same as training model)
    """
    
    def __init__(self, state_size, action_size, hidden_size=512):
        super(WinstonAI, self).__init__()
        
        # Feature extraction layers
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
        
        # LSTM for sequential pattern recognition
        self.lstm = nn.LSTM(
            input_size=hidden_size // 4,
            hidden_size=hidden_size // 8,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Decision layers
        self.decision_layers = nn.Sequential(
            nn.Linear(hidden_size // 8, hidden_size // 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 16, action_size)
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Reshape for LSTM
        features = features.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]
        
        # Final decision
        q_values = self.decision_layers(lstm_out)
        
        return q_values

class WinstonAITrader:
    """
    Live trading bot using WinstonAI model with PocketOption
    """
    
    def __init__(self, model_path, config_file="trading_config.json"):
        self.config = self.load_config(config_file)
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback_window = 50
        self.candle_buffer = deque(maxlen=100)  # Store recent candles
        self.trade_history = []
        self.balance = 0
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=1)  # Minimum time between trades
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.load_model(model_path)
        
        # Setup logging
        self.setup_logging()
        
    def load_config(self, config_file):
        """Load trading configuration"""
        default_config = {
            "ssid": "",
            "assets": ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"],
            "trade_amount": 1.0,
            "trade_duration": 60,
            "max_trades_per_hour": 10,
            "min_confidence": 0.7,
            "risk_management": {
                "max_daily_loss": 100.0,
                "max_consecutive_losses": 3,
                "stop_loss_percentage": 0.1
            },
            "trading_hours": {
                "start": "08:00",
                "end": "18:00"
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Creating default config...")
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    
    def setup_logging(self):
        """Setup logging for the trading bot"""
        # Create formatters (removed emojis to avoid encoding issues)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create handlers
        file_handler = logging.FileHandler('winston_ai_trading.log', encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def load_model(self, model_path):
        """Load the trained WinstonAI model"""
        try:
            # Calculate state size based on number of indicators
            # Use sufficient sample data for all indicators to work properly
            sample_data = pd.DataFrame({
                'open': [1.0] * 100,
                'high': [1.1] * 100,
                'close': [1.0] * 100,
                'low': [0.9] * 100
            })
            sample_indicators = TechnicalIndicators.calculate_indicators(sample_data)
            feature_columns = [col for col in sample_indicators.columns if col not in ['time', 'asset']]
            state_size = len(feature_columns) * self.lookback_window
            
            # Initialize model
            self.model = WinstonAI(state_size=state_size, action_size=3).to(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['q_network'])
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            self.logger.info(f"State size: {state_size}, Actions: 3 (Hold, Call, Put)")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_state(self, candles_df):
        """Prepare state from candle data for model prediction"""
        try:
            if len(candles_df) < self.lookback_window:
                self.logger.warning(f"Not enough candle data: {len(candles_df)} < {self.lookback_window}")
                return None
            
            # Calculate technical indicators
            df_with_indicators = TechnicalIndicators.calculate_indicators(candles_df)
            
            # Get feature columns (exclude metadata)
            feature_columns = [col for col in df_with_indicators.columns 
                             if col not in ['time', 'asset']]
            
            # Check for valid data after indicators
            if df_with_indicators[feature_columns].isnull().all().any():
                self.logger.warning("Some indicators returned all NaN values")
                return None
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(df_with_indicators[feature_columns])
            
            # Get last lookback_window rows and flatten
            state = scaled_data[-self.lookback_window:].flatten()
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error preparing state: {e}")
            return None
    
    def predict_action(self, state):
        """Predict action using WinstonAI model"""
        try:
            if state is None:
                return 0, 0.0  # Hold action with 0 confidence
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.model(state_tensor)
                
                # Get action and confidence
                action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                confidence = torch.softmax(q_values, dim=1).max().cpu().numpy()
                
                return action, confidence
                
        except Exception as e:
            self.logger.error(f"Error predicting action: {e}")
            return 0, 0.0
    
    def is_trading_time(self):
        """Check if current time is within trading hours"""
        now = datetime.now().time()
        start_time = datetime.strptime(self.config["trading_hours"]["start"], "%H:%M").time()
        end_time = datetime.strptime(self.config["trading_hours"]["end"], "%H:%M").time()
        
        return start_time <= now <= end_time
    
    def can_trade(self):
        """Check if bot can place a new trade based on risk management"""
        # Check trading hours
        if not self.is_trading_time():
            return False, "Outside trading hours"
        
        # Check minimum time between trades
        if (self.last_trade_time and 
            datetime.now() - self.last_trade_time < self.min_trade_interval):
            return False, "Too soon for next trade"
        
        # Check daily loss limit
        today_trades = [t for t in self.trade_history 
                       if t['timestamp'].date() == datetime.now().date()]
        daily_pnl = sum(t['pnl'] for t in today_trades)
        
        if daily_pnl < -self.config["risk_management"]["max_daily_loss"]:
            return False, "Daily loss limit reached"
        
        # Check consecutive losses
        recent_trades = sorted(self.trade_history, key=lambda x: x['timestamp'], reverse=True)
        consecutive_losses = 0
        for trade in recent_trades:
            if trade['pnl'] < 0:
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= self.config["risk_management"]["max_consecutive_losses"]:
            return False, "Too many consecutive losses"
        
        return True, "OK"
    
    async def execute_trade(self, api, asset, action, confidence):
        """Execute a trade based on model prediction"""
        try:
            can_trade_result, reason = self.can_trade()
            if not can_trade_result:
                self.logger.info(f"Cannot trade: {reason}")
                return None
            
            # Check confidence threshold
            if confidence < self.config["min_confidence"]:
                self.logger.info(f"Low confidence {confidence:.3f}, skipping trade")
                return None
            
            trade_amount = self.config["trade_amount"]
            trade_duration = self.config["trade_duration"]
            
            # Execute trade based on action
            if action == 1:  # Call option
                self.logger.info(f"Placing CALL trade: {asset}, Amount: ${trade_amount}, Duration: {trade_duration}s, Confidence: {confidence:.3f}")
                trade_id, trade_data = await api.buy(
                    asset=asset, 
                    amount=trade_amount, 
                    time=trade_duration, 
                    check_win=False
                )
                trade_type = "CALL"
                
            elif action == 2:  # Put option
                self.logger.info(f"Placing PUT trade: {asset}, Amount: ${trade_amount}, Duration: {trade_duration}s, Confidence: {confidence:.3f}")
                trade_id, trade_data = await api.sell(
                    asset=asset, 
                    amount=trade_amount, 
                    time=trade_duration, 
                    check_win=False
                )
                trade_type = "PUT"
                
            else:  # Hold
                self.logger.info(f"Model suggests HOLD for {asset}")
                return None
            
            # Record trade
            trade_record = {
                'trade_id': trade_id,
                'asset': asset,
                'type': trade_type,
                'amount': trade_amount,
                'duration': trade_duration,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'status': 'open'
            }
            
            self.trade_history.append(trade_record)
            self.last_trade_time = datetime.now()
            
            self.logger.info(f"Trade placed successfully: ID {trade_id}")
            
            # Schedule trade result check
            asyncio.create_task(self.check_trade_result(api, trade_id, trade_duration))
            
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    async def check_trade_result(self, api, trade_id, duration):
        """Check trade result after expiration"""
        try:
            # Wait for trade to expire plus buffer
            await asyncio.sleep(duration + 10)
            
            # Check win result
            trade_result = await api.check_win(trade_id)
            
            # Update trade history
            for trade in self.trade_history:
                if trade['trade_id'] == trade_id:
                    trade['result'] = trade_result['result']
                    trade['pnl'] = trade_result.get('pnl', 0)
                    trade['status'] = 'closed'
                    
                    if trade_result['result'] == 'win':
                        self.logger.info(f"Trade {trade_id} WON! PnL: +${trade['pnl']:.2f}")
                    else:
                        self.logger.info(f"Trade {trade_id} LOST! PnL: -${trade['amount']:.2f}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error checking trade result for {trade_id}: {e}")
    
    async def monitor_asset(self, api, asset):
        """Monitor a specific asset and trade when conditions are met"""
        self.logger.info(f"Starting monitoring for {asset}")
        
        try:
            # Get initial candle data using correct API method
            candles = await api.get_candles(asset, 60, 6000)  # Last 100 1-minute candles (100 * 60 seconds)
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Store in buffer
            for _, candle in df.iterrows():
                self.candle_buffer.append(candle.to_dict())
            
            self.logger.info(f"Loaded {len(candles)} historical candles for {asset}")
            
            # Subscribe to real-time data
            stream = await api.subscribe_symbol(asset)
            async for candle_data in stream:
                try:
                    # Add new candle to buffer
                    self.candle_buffer.append(candle_data)
                    
                    # Convert buffer to DataFrame
                    candles_df = pd.DataFrame(list(self.candle_buffer))
                    
                    # Prepare state for model
                    state = self.prepare_state(candles_df)
                    
                    if state is not None:
                        # Get model prediction
                        action, confidence = self.predict_action(state)
                        
                        # Log prediction
                        action_names = ["HOLD", "CALL", "PUT"]
                        self.logger.info(f"{asset}: Model predicts {action_names[action]} (confidence: {confidence:.3f})")
                        
                        # Execute trade if action is not hold
                        if action != 0:
                            await self.execute_trade(api, asset, action, confidence)
                    
                    # Small delay to prevent excessive processing
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing candle for {asset}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error monitoring asset {asset}: {e}")
    
    async def run_trading_bot(self):
        """Main trading bot loop"""
        self.logger.info("Starting WinstonAI Live Trading Bot")
        self.logger.info("=" * 60)
        
        # Check if SSID is provided
        if not self.config["ssid"]:
            ssid = input("Please enter your PocketOption SSID: ").strip()
            self.config["ssid"] = ssid
            # Save updated config
            with open("trading_config.json", 'w') as f:
                json.dump(self.config, f, indent=4)
        
        try:
            # Initialize API
            api = PocketOptionAsync(self.config["ssid"])
            await asyncio.sleep(5)  # Wait for connection
            
            # Get account balance
            balance_info = await api.balance()
            balance_data = json.loads(balance_info)
            self.balance = balance_data.get('balance', 0)
            self.logger.info(f"Account Balance: ${self.balance:.2f}")
            
            # Start monitoring multiple assets
            tasks = []
            for asset in self.config["assets"]:
                task = asyncio.create_task(self.monitor_asset(api, asset))
                tasks.append(task)
            
            # Run all monitoring tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Error in main trading loop: {e}")
        
        self.logger.info("Trading bot stopped")
    
    def print_trading_summary(self):
        """Print trading session summary"""
        if not self.trade_history:
            print("No trades executed in this session.")
            return
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.get('result') == 'win')
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl', -t.get('amount', 0)) for t in self.trade_history)
        
        print("\n" + "="*50)
        print("WINSTON AI TRADING SUMMARY")
        print("="*50)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Final Balance: ${self.balance:.2f}")
        print("="*50)

async def main():
    """Main function"""
    # Setup logging for the library
    start_logs(path=".", level="INFO", terminal=True)
    
    # Initialize WinstonAI trader with the latest trained model
    trader = WinstonAITrader(
        model_path=r"c:\Users\tp\ComunityPrograms\WinstonAI\winston_ai_episode_500.pth",  # Use absolute path
        config_file="trading_config.json"
    )
    
    try:
        # Run the trading bot
        await trader.run_trading_bot()
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    finally:
        # Print summary
        trader.print_trading_summary()

if __name__ == "__main__":
    print("WinstonAI Live Trading Bot")
    print("Press Ctrl+C to stop the bot")
    print("Check winston_ai_trading.log for detailed logs")
    print("-" * 50)
    
    # Run the bot
    asyncio.run(main())
