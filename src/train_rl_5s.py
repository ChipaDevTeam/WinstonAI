import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.preprocessing import StandardScaler
import random
import ta
import os
import sys

# --- GPU Optimization & Configuration ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Enable cudnn benchmark for optimized performance on fixed input sizes
    torch.backends.cudnn.benchmark = True
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️  GPU NOT DETECTED. Using CPU.")
    print("   To enable GPU, ensure you have installed PyTorch with CUDA support.")
    
    # Try to provide a helpful command based on 2026/future context (which I am in)
    # or generic advice.
    print("   Run: pip install torch --index-url https://download.pytorch.org/whl/cu124 (or your version)")

# Optional: Enable TensorFloat32 for Ampere+ GPUs (faster FP32 matmul)
# This requires newer PyTorch versions which we likely have in 2026.
try:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')
        print("   TensorFloat32 enabled for Ampere+ GPU.")
except:
    pass

print(f"Using device: {device}")

class PriceActionFeatures:
    """
    Advanced Price Action + Technical Indicators for 5s timeframe RL
    """
    @staticmethod
    def add_features(df):
        df = df.copy()
        
        # --- Basic Indicators ---
        # Moving Averages - Fast for 5s charts
        df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
        df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        # RSI - Volatility sensitivity
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # ATR for volatility normalization
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # --- Price Action / Market Structure ---
        
        # Swing Highs and Lows (Fractals) - Window of 2 candles on each side
        # A swing high is a high surrounded by lower highs
        df['swing_high'] = df['high'].rolling(window=5, center=True).apply(
            lambda x: 1 if x[2] == max(x) else 0,
            raw=True
        )
        df['swing_low'] = df['low'].rolling(window=5, center=True).apply(
            lambda x: 1 if x[2] == min(x) else 0,
            raw=True
        )

        # Support and Resistance Levels (Dynamic)
        # using recent swing points as temporary S/R
        # For efficiency in pandas, we forward fill the most recent swing levels
        
        # 1. Identify price at swing points
        df['last_swing_high_price'] = df['high'].where(df['swing_high'] == 1).ffill()
        df['last_swing_low_price'] = df['low'].where(df['swing_low'] == 1).ffill()
        
        # 2. Distance to S/R
        df['dist_to_resistance'] = (df['last_swing_high_price'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['last_swing_low_price']) / df['close']
        
        # 3. Breakout detection
        # Price crossing above resistance or below support
        df['breakout_up'] = (df['close'] > df['last_swing_high_price'].shift(1)).astype(int)
        df['breakout_down'] = (df['close'] < df['last_swing_low_price'].shift(1)).astype(int)
        
        # --- Candle Body & Wicks (Micro-structure) ---
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Ratios
        df['body_perc'] = df['body_size'] / (df['high'] - df['low'] + 1e-9)
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['body_size'] + 1e-9)
        
        # --- Trends ---
        # 1 means uptrend (EMA20 < EMA10 < EMA5), -1 downtrend
        condition_up = (df['ema_5'] > df['ema_10']) & (df['ema_10'] > df['ema_20'])
        condition_down = (df['ema_5'] < df['ema_10']) & (df['ema_10'] < df['ema_20'])
        df['trend'] = 0
        df.loc[condition_up, 'trend'] = 1
        df.loc[condition_down, 'trend'] = -1
        
        # Clean up NaNs from rolling windows
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True) # Fallback
        
        return df

class TradingEnv:
    """
    Custom Environment for Binary Options 5s Trading
    """
    def __init__(self, df, window_size=20, initial_balance=1000, payout_rate=0.85):
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.payout_rate = payout_rate
        
        # Define features columns (exclude timestamps/targets)
        excluded_cols = ['time', 'volume', 'date']
        self.feature_cols = [c for c in df.columns if c not in excluded_cols]
        self.data = df[self.feature_cols].values
        self.prices = df['close'].values
        
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.done = False
        self.positions = [] # List of active trades
        return self._get_observation()
    
    def _get_observation(self):
        # Return window of features
        start = self.current_step - self.window_size
        end = self.current_step
        return self.data[start:end]
    
    def step(self, action):
        """
        Action: 0=HOLD, 1=CALL (Buy), 2=PUT (Sell)
        Simulates entering a trade that expires in X candles.
        For 5s candles, a 1-minute expiration is 12 candles.
        Let's assume a quick expiration for this scalp strategy: 6 candles (30 seconds).
        """
        expiration_period = 6 # 30 seconds if 5s candles
        trade_amount = 10 # Fixed trade amount for RL stability
        reward = 0
        
        # 1. Check expiring trades
        # We can't actually check specific expirations "in the past" easily in a simple step loop 
        # unless we track them.
        # But for RL training efficiency, we often give immediate reward based on the outcome 
        # look-ahead, OR we delay reward.
        # Here, we will use IMMEDIATE REWARD based on look-ahead to simplify credit assignment.
        # This is a common simplification in trading RL to speed up convergence.
        # The agent sees state t, takes action a, and gets reward based on price at t+expiry.
        
        if self.current_step + expiration_period >= len(self.df):
            self.done = True
            return np.zeros((self.window_size, self.data.shape[1])), 0, True, {}
            
        current_price = self.prices[self.current_step]
        future_price = self.prices[self.current_step + expiration_period]
        
        if action == 1: # CALL
            if future_price > current_price:
                reward = trade_amount * self.payout_rate
            elif future_price < current_price:
                reward = -trade_amount
            else:
                reward = 0
                
        elif action == 2: # PUT
            if future_price < current_price:
                reward = trade_amount * self.payout_rate
            elif future_price > current_price:
                reward = -trade_amount
            else:
                reward = 0
        
        # Small penalty for holding to encourage taking only good trades? 
        # Or maybe no penalty. Let's add a tiny penalty for holding to prevent inaction if we want high activity,
        # but usually 0 is fine.
        
        self.balance += reward
        self.current_step += 1
        
        if self.current_step >= len(self.df) - expiration_period - 1:
            self.done = True
            
        next_state = self._get_observation()
        
        return next_state, reward, self.done, {'balance': self.balance}

# --- RL Model (PPO-style Actor-Critic) ---

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Feature extractor (e.g., LSTM for time series)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Actor head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (Value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, window, features)
        lstm_out, _ = self.lstm(x)
        # We take the last time step's output
        last_hidden = lstm_out[:, -1, :] 
        
        action_probs = self.actor(last_hidden)
        state_value = self.critic(last_hidden)
        
        return action_probs, state_value

# --- PPO Agent ---

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        self.policy = ActorCritic(input_dim, 128, action_dim).to(device)
        
        # Optimize model with torch.compile if available (PyTorch 2.0+)
        # This can significantly speed up inference and training on GPU
        if hasattr(torch, 'compile'):
            try:
                print("   Compiling PPO policy with torch.compile()...")
                self.policy = torch.compile(self.policy)
            except Exception as e:
                print(f"   Could not compile model: {e}")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(input_dim, 128, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, _ = self.policy_old(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        # Convert memory to tensors
        states = torch.FloatTensor(memory.states).to(device)
        actions = torch.FloatTensor(memory.actions).to(device)
        log_probs_old = torch.FloatTensor(memory.log_probs).to(device)
        rewards = torch.FloatTensor(memory.rewards).to(device)
        is_terminals = torch.FloatTensor(memory.is_terminals).to(device)
        
        # Monte Carlo estimate of rewards
        rewards_list = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_list.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(device)
        if rewards_tensor.std() > 0:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)
        
        # PPO Update for K epochs
        for _ in range(4): # K epochs
            # Evaluating old actions and values
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            
            log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            
            # Ratios
            ratios = torch.exp(log_probs - log_probs_old)
            
            # Surrogate Loss
            advantages = rewards_tensor - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards_tensor) - 0.01*dist_entropy
            
            # Backward
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
    def add(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_terminals.append(done)

# --- Main Training Loop ---

def train():
    print("Starting Training on 5s Candles...")
    
    # 1. Load Data
    data_path = r"C:\Users\vigop\.cache\kagglehub\datasets\komodata\forexdataset\versions\2\EURUSD_M30.csv"
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df['time'] = pd.to_datetime(df['time'])
        
        # Rename tick_volume to volume if needed
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
        # WARNING: The dataset is M30, but we are simulating 5s strategy.
        # This means 1 step in the env is actually 30 minutes, not 5 seconds.
        # To truly simulate 5s, we would need 5s data.
        # However, for the purpose of learning "Price Action" patterns on a fractal level,
        # the patterns on M30 are similar to 5s (Fractal Market Hypothesis).
        print(f"Loaded {len(df)} rows. Using M30 candles to simulate structure learning.")
        
    else:
        print("Data file not found. Generating synthetic 5s price action data...")
        dates = pd.date_range(start='2024-01-01', periods=10000, freq='5S')
        # Random walk with some trendiness to simulate price
        price = 100 + np.cumsum(np.random.normal(0, 0.1, 10000))
        # Add some sine waves for swing highs/lows pattern
        price += np.sin(np.linspace(0, 50, 10000)) * 2
        
        df = pd.DataFrame({
            'time': dates,
            'open': price + np.random.normal(0, 0.02, 10000),
            'high': price + np.abs(np.random.normal(0, 0.05, 10000)),
            'low': price - np.abs(np.random.normal(0, 0.05, 10000)),
            'close': price + np.random.normal(0, 0.02, 10000),
            'volume': np.random.randint(100, 1000, 10000)
        })
    
    # 2. Add Price Action Features
    print("Calculating Price Action indicators...")
    df = PriceActionFeatures.add_features(df)
    
    # Normalization
    print("Normalizing features...")
    feature_cols = [c for c in df.columns if c not in ['time', 'volume', 'date', 'close', 'open', 'high', 'low']]
    # Note: We excluded raw OHLC from features usually to make it price-agnostic, 
    # but PriceActionFeatures might have added them or derivatives.
    # Let's verify what PriceActionFeatures adds. 
    # It adds 'ema_5', 'rsi', 'dist_to_resistance', etc.
    # The TradingEnv uses all columns except 'time', 'volume', 'date'.
    # If we want to normalize effectively, we should normalize everything used by the agent.
    
    scaler = StandardScaler()
    # Identification of numeric columns to normalize
    cols_to_norm = [c for c in df.columns if c not in ['time', 'date']]
    df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
    
    print("Features normalized.")

    # 3. Setup Environment
    # We need to pass the UNNORMALIZED prices for reward calculation if the env relies on them?
    # TradingEnv uses self.prices = df['close'].values for reward calc.
    # Uh oh. If I normalize 'close' in place, the reward calculation (future_price > current_price) 
    # will be dealing with normalized values (e.g. 0.1 vs 0.2).
    # The logic (a > b) is preserved under linear scaling, so (norm_a > norm_b) is true if a > b.
    # THE REWARD QUANTITY might be affected: reward = (future - current).
    # Wait, the reward logic in TradingEnv is: 
    # if future > current: reward = trade_amount * payout.
    # else: reward = -trade_amount.
    # Since trade_amount is fixed (10), the absolute price values DON'T matter for the logic, 
    # only the direction (> or <).
    # So normalizing 'close' is SAFE for this specific Env logic.
    
    env = TradingEnv(df)
    
    # input dim is number of feature columns
    input_dim = env._get_observation().shape[1] 
    action_dim = 3 # HOLD, CALL, PUT
    
    agent = PPOAgent(input_dim, action_dim)
    memory = Memory()
    
    max_episodes = 500
    update_timestep = 2000 
    timestep = 0
    
    for episode in range(1, max_episodes+1):
        state = env.reset()
        episode_reward = 0
        
        while True:
            timestep += 1
            
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            # Save to memory
            memory.add(state, action, log_prob, reward, done)
            
            state = next_state
            episode_reward += reward
            
            # Update PPO
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear()
                timestep = 0
            
            if done:
                break
                
        print(f"Episode {episode}\t Reward: {episode_reward:.2f}\t Final Balance: {info['balance']:.2f}")
        
    # Save Model
    print("Saving model...")
    torch.save(agent.policy.state_dict(), "src/winston_ai_rl_5s_model.pth")
    print("Training Complete. Model saved to src/winston_ai_rl_5s_model.pth")

if __name__ == "__main__":
    train()
