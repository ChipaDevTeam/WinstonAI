"""
Binary options trading environment for reinforcement learning
"""

import pandas as pd
import numpy as np
from winston_ai.indicators.technical import TechnicalIndicators


class BinaryOptionsEnvironment:
    """
    Binary options trading environment for reinforcement learning
    
    This environment simulates binary options trading with configurable
    parameters and provides state, action, and reward interfaces for
    training RL agents.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        lookback_window: int = 100,
        trade_duration: int = 60,
        initial_balance: float = 10000,
        trade_amount: float = 100,
        payout_ratio: float = 0.8
    ):
        """
        Initialize trading environment
        
        Args:
            data: DataFrame with OHLCV price data
            lookback_window: Number of historical steps to include in state
            trade_duration: Number of steps for each trade to expire
            initial_balance: Starting account balance
            trade_amount: Amount to invest per trade
            payout_ratio: Profit ratio for winning trades (0.8 = 80% profit)
        """
        self.data = data.copy()
        self.lookback_window = lookback_window
        self.trade_duration = trade_duration
        self.action_space = 3  # Hold (0), Call (1), Put (2)
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.trade_amount = trade_amount
        self.payout_ratio = payout_ratio
        
        # Calculate technical indicators
        self.indicators = TechnicalIndicators.calculate_all_indicators(self.data)
        
        # Combine price data with indicators
        self.features = pd.concat([
            self.data[['open', 'high', 'low', 'close']],
            self.indicators
        ], axis=1)
        
        self.state_size = len(self.features.columns) * lookback_window
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
            Initial state observation
        """
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_history = []
        self.open_trades = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state observation
        
        Returns:
            State vector containing normalized features
        """
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            state = np.zeros(self.state_size)
        else:
            # Get lookback window of features
            window_data = self.features.iloc[
                self.current_step - self.lookback_window:self.current_step
            ].values
            
            # Normalize features
            window_data = (window_data - np.mean(window_data, axis=0)) / (np.std(window_data, axis=0) + 1e-8)
            
            # Flatten to 1D state vector
            state = window_data.flatten()
        
        return state
    
    def step(self, action: int) -> tuple:
        """
        Execute action and step environment forward
        
        Args:
            action: Action to take (0=Hold, 1=Call, 2=Put)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Process open trades
        reward = self._process_open_trades()
        
        # Execute new action
        if action == 1:  # Call
            reward += self._execute_trade("call")
        elif action == 2:  # Put
            reward += self._execute_trade("put")
        # action == 0 is hold, no trade executed
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1) or (self.balance <= 0)
        
        next_state = self._get_state() if not done else np.zeros(self.state_size)
        
        info = self.get_stats()
        
        return next_state, reward, done, info
    
    def _execute_trade(self, trade_type: str) -> float:
        """
        Execute a binary options trade
        
        Args:
            trade_type: Type of trade ('call' or 'put')
            
        Returns:
            Immediate reward (penalty if insufficient funds)
        """
        if self.balance < self.trade_amount:
            return -10  # Penalty for insufficient funds
        
        current_price = self.data.iloc[self.current_step]['close']
        
        trade = {
            'type': trade_type,
            'entry_price': current_price,
            'entry_time': self.current_step,
            'expiry_time': self.current_step + self.trade_duration,
            'amount': self.trade_amount
        }
        
        self.open_trades.append(trade)
        self.balance -= self.trade_amount
        
        return 0  # No immediate reward, wait for trade completion
    
    def _process_open_trades(self) -> float:
        """
        Process and close expired trades
        
        Returns:
            Total reward from closed trades
        """
        total_reward = 0
        trades_to_remove = []
        
        for i, trade in enumerate(self.open_trades):
            if self.current_step >= trade['expiry_time']:
                # Trade has expired, determine outcome
                if trade['expiry_time'] < len(self.data):
                    expiry_price = self.data.iloc[trade['expiry_time']]['close']
                    
                    # Determine if trade won
                    if trade['type'] == 'call':
                        won = expiry_price > trade['entry_price']
                    else:  # put
                        won = expiry_price < trade['entry_price']
                    
                    # Calculate reward
                    if won:
                        payout = trade['amount'] * (1 + self.payout_ratio)
                        self.balance += payout
                        self.winning_trades += 1
                        reward = payout - trade['amount']
                    else:
                        reward = -trade['amount']
                    
                    total_reward += reward
                    self.total_trades += 1
                    self.trade_history.append({
                        'trade': trade,
                        'won': won,
                        'reward': reward
                    })
                
                trades_to_remove.append(i)
        
        # Remove processed trades
        for i in reversed(trades_to_remove):
            self.open_trades.pop(i)
        
        return total_reward
    
    def get_stats(self) -> dict:
        """
        Get trading statistics
        
        Returns:
            Dictionary containing trading performance metrics
        """
        total_profit = self.balance - self.initial_balance
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'final_balance': self.balance,
            'roi': (total_profit / self.initial_balance) * 100,
            'current_step': self.current_step,
            'open_trades': len(self.open_trades)
        }
