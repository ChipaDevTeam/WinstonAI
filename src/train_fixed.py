# WinstonAI - Binary Options Trading AI
# Comprehensive reinforcement learning model for binary options trading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import os
import glob
import re
import subprocess
import sys
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
        
        # MFI (Money Flow Index) - Skip since we don't have volume data
        # Using a simpler price-based momentum indicator instead
        df['price_momentum'] = df['close'].rolling(window=14).mean() / df['close'].rolling(window=28).mean()
        
        # Price Action Features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # Fibonacci Levels (calculate from recent high/low)
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

class BinaryOptionsEnvironment:
    """
    Trading environment for binary options
    """
    
    def __init__(self, data, lookback_window=50, trade_duration=60):
        self.data = data
        self.lookback_window = lookback_window
        self.trade_duration = trade_duration
        self.current_step = lookback_window
        self.max_steps = len(data) - trade_duration
        self.balance = 10000  # Starting balance
        self.initial_balance = self.balance
        self.trades_history = []
        
        # Action space: 0=Hold, 1=Call, 2=Put
        self.action_space = 3
        
        # Feature engineering
        feature_columns = [col for col in data.columns if col not in ['time', 'asset']]
        self.scaler = MinMaxScaler()
        self.scaled_data = pd.DataFrame(
            self.scaler.fit_transform(data[feature_columns]),
            columns=feature_columns,
            index=data.index
        )
        
        self.state_size = len(feature_columns) * lookback_window
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.trades_history = []
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        if self.current_step < self.lookback_window:
            return np.zeros(self.state_size)
        
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        state_data = self.scaled_data.iloc[start_idx:end_idx].values
        return state_data.flatten()
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate future price for binary option outcome
        future_step = min(self.current_step + self.trade_duration, len(self.data) - 1)
        future_price = self.data.iloc[future_step]['close']
        
        reward = 0
        trade_amount = 100  # Fixed trade amount
        
        if action == 1:  # Call option
            if future_price > current_price:
                reward = 0.8 * trade_amount  # 80% profit on winning trade
                self.balance += reward
            else:
                reward = -trade_amount  # Lose trade amount
                self.balance += reward
                
        elif action == 2:  # Put option
            if future_price < current_price:
                reward = 0.8 * trade_amount  # 80% profit on winning trade
                self.balance += reward
            else:
                reward = -trade_amount  # Lose trade amount
                self.balance += reward
        
        # Record trade
        if action != 0:
            trade_info = {
                'step': self.current_step,
                'action': action,
                'entry_price': current_price,
                'exit_price': future_price,
                'reward': reward,
                'balance': self.balance
            }
            self.trades_history.append(trade_info)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self.get_state(), reward, done
    
    def get_stats(self):
        """Get trading statistics"""
        if not self.trades_history:
            return {}
        
        total_trades = len(self.trades_history)
        winning_trades = sum(1 for trade in self.trades_history if trade['reward'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = self.balance - self.initial_balance
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'final_balance': self.balance,
            'roi': (total_profit / self.initial_balance) * 100
        }

class WinstonAI(nn.Module):
    """
    Deep Q-Network for binary options trading
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
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Reshape for LSTM (assuming single timestep)
        features = features.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        
        # Final decision
        q_values = self.decision_layers(lstm_out)
        
        return q_values

class DQNAgent:
    """
    Deep Q-Learning Agent for trading
    """
    
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.batch_size = 32
        self.gamma = 0.95
        
        # Neural networks
        self.q_network = WinstonAI(state_size, action_size).to(device)
        self.target_network = WinstonAI(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath, episode=None, metrics=None):
        """Save model with training state"""
        save_dict = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        
        if episode is not None:
            save_dict['episode'] = episode
            
        if metrics is not None:
            save_dict['metrics'] = metrics
            
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        """Load model with training state"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        
        return checkpoint.get('episode', 0), checkpoint.get('metrics', {})

def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    checkpoint_pattern = "winston_ai_episode_*.pth"
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None, 0
    
    # Extract episode numbers from filenames
    episodes = []
    for file in checkpoint_files:
        match = re.search(r'episode_(\d+)\.pth', file)
        if match:
            episodes.append((int(match.group(1)), file))
    
    if not episodes:
        return None, 0
    
    # Find the latest episode
    latest_episode, latest_file = max(episodes, key=lambda x: x[0])
    print(f"Found latest checkpoint: {latest_file} (Episode {latest_episode})")
    
    return latest_file, latest_episode

def load_and_prepare_data(filepath):
    """Load and prepare data for training"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Convert time column to datetime with ISO8601 format
    df['time'] = pd.to_datetime(df['time'], format='ISO8601')
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # For now, use data from one asset for training
    # You can modify this to use multiple assets
    assets = df['asset'].unique()
    print(f"Available assets: {assets}")
    
    # Use the first asset with sufficient data
    for asset in assets:
        asset_data = df[df['asset'] == asset].copy()
        if len(asset_data) > 1000:  # Minimum data requirement
            print(f"Using asset: {asset} with {len(asset_data)} data points")
            df = asset_data
            break
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    df = TechnicalIndicators.calculate_indicators(df)
    
    # Check for NaN values before cleaning
    print(f"NaN counts before cleaning:")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0])
    
    # Remove rows with NaN values, but keep a minimum number of rows
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    final_rows = len(df)
    
    if final_rows < 100:  # Need minimum data for training
        print(f"Warning: Only {final_rows} rows remain after cleaning. Using forward fill for NaN values.")
        # Reload data and use forward fill instead
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'], format='ISO8601')
        df = df.sort_values('time').reset_index(drop=True)
        
        for asset in assets:
            asset_data = df[df['asset'] == asset].copy()
            if len(asset_data) > 1000:
                df = asset_data
                break
        
        df = TechnicalIndicators.calculate_indicators(df)
        # Forward fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    
    print(f"Final dataset shape: {df.shape}")
    return df

def train_winston_ai():
    """Main training function"""
    print("🤖 Initializing WinstonAI - Binary Options Trading AI")
    print("=" * 60)
    
    # Load and prepare data
    data_path = r"c:\Users\tp\ComunityPrograms\all_assets_candles.csv"
    df = load_and_prepare_data(data_path)
    
    # Create environment
    env = BinaryOptionsEnvironment(df, lookback_window=50, trade_duration=60)
    
    # Create agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_space,
        lr=0.001
    )
    
    # Check for existing checkpoints and load the latest one
    latest_checkpoint, start_episode = find_latest_checkpoint()
    
    # Training metrics (initialize before loading checkpoint)
    episode_rewards = []
    episode_profits = []
    win_rates = []
    
    if latest_checkpoint:
        print(f"🔄 Loading checkpoint: {latest_checkpoint}")
        loaded_episode, loaded_metrics = agent.load(latest_checkpoint)
        start_episode = loaded_episode
        
        # Load previous metrics if available
        if loaded_metrics:
            episode_rewards = loaded_metrics.get('rewards', [])
            episode_profits = loaded_metrics.get('profits', [])
            win_rates = loaded_metrics.get('win_rates', [])
            print(f"📈 Loaded {len(episode_rewards)} previous training metrics")
        
        print(f"✅ Resumed training from episode {start_episode}")
    else:
        print("🆕 Starting training from scratch")
        start_episode = 0
    
    # Training parameters
    total_episodes = 1000
    target_update_freq = 10
    save_freq = 100
    
    print(f"Starting training from episode {start_episode} to {total_episodes}...")
    print(f"State size: {env.state_size}")
    print(f"Action space: {env.action_space}")
    print(f"Using device: {device}")
    print("=" * 60)
    
    for episode in range(start_episode, total_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Train the agent
        agent.replay()
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Get episode statistics
        stats = env.get_stats()
        episode_rewards.append(total_reward)
        episode_profits.append(stats.get('total_profit', 0))
        win_rates.append(stats.get('win_rate', 0))
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            avg_profit = np.mean(episode_profits[-50:]) if episode_profits else 0
            avg_win_rate = np.mean(win_rates[-50:]) if win_rates else 0
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:8.2f} | "
                  f"Profit: {stats.get('total_profit', 0):8.2f} | "
                  f"Win Rate: {stats.get('win_rate', 0)*100:5.1f}% | "
                  f"Trades: {stats.get('total_trades', 0):3d} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
            print(f"         | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Profit: {avg_profit:7.2f} | "
                  f"Avg Win Rate: {avg_win_rate*100:5.1f}%")
            print("-" * 80)
        
        # Save model periodically
        if episode % save_freq == 0 and episode > start_episode:
            model_path = f"winston_ai_episode_{episode}.pth"
            metrics = {
                'rewards': episode_rewards,
                'profits': episode_profits,
                'win_rates': win_rates
            }
            agent.save(model_path, episode=episode, metrics=metrics)
            print(f"💾 Model saved: {model_path}")
    
    # Final save
    final_metrics = {
        'rewards': episode_rewards,
        'profits': episode_profits,
        'win_rates': win_rates
    }
    agent.save("winston_ai_final.pth", episode=total_episodes-1, metrics=final_metrics)
    print("\n🎉 Training completed!")
    print(f"📊 Final model saved as: winston_ai_final.pth")
    
    # Plot training results
    plot_training_results(episode_rewards, episode_profits, win_rates)
    
    return agent, env

def plot_training_results(rewards, profits, win_rates):
    """Plot training metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Episode profits
    ax2.plot(profits)
    ax2.set_title('Episode Profits')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Profit ($)')
    ax2.grid(True)
    
    # Win rates
    ax3.plot([w * 100 for w in win_rates])
    ax3.set_title('Win Rates')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Win Rate (%)')
    ax3.grid(True)
    
    # Moving averages
    window = 50
    if len(rewards) >= window:
        moving_avg_rewards = pd.Series(rewards).rolling(window=window).mean()
        moving_avg_profits = pd.Series(profits).rolling(window=window).mean()
        
        ax4.plot(moving_avg_rewards, label='Avg Rewards')
        ax4.plot(moving_avg_profits, label='Avg Profits')
        ax4.set_title(f'Moving Averages (Window: {window})')
        ax4.set_xlabel('Episode')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('winston_ai_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Training plots saved as: winston_ai_training_results.png")

def test_winston_ai(model_path):
    """Test the trained model"""
    print("🧪 Testing WinstonAI...")
    
    # Load data
    data_path = r"c:\Users\tp\ComunityPrograms\all_assets_candles.csv"
    df = load_and_prepare_data(data_path)
    
    # Create environment
    env = BinaryOptionsEnvironment(df, lookback_window=50, trade_duration=60)
    
    # Create and load agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_space
    )
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during testing
    
    # Run test episode
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.act(state, training=False)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Print results
    stats = env.get_stats()
    print(f"Test Results:")
    print(f"Total Reward: ${total_reward:.2f}")
    print(f"Total Profit: ${stats['total_profit']:.2f}")
    print(f"Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"ROI: {stats['roi']:.2f}%")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import ta
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ta", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
    
    # Train the model
    agent, env = train_winston_ai()
    
    # Test the final model
    test_winston_ai("winston_ai_final.pth")
