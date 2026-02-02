"""
High-level trainer for WinstonAI models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Optional, Dict, List

from winston_ai.training.environment import BinaryOptionsEnvironment
from winston_ai.training.agent import DQNAgent
from winston_ai.utils.config import Config
from winston_ai.utils.checkpoints import save_checkpoint, find_latest_checkpoint


class Trainer:
    """
    High-level trainer for WinstonAI reinforcement learning models
    
    This class orchestrates the training process including:
    - Environment setup
    - Agent training
    - Checkpoint management
    - Metrics tracking and visualization
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[Config] = None,
        checkpoint_dir: str = "models"
    ):
        """
        Initialize trainer
        
        Args:
            data: Training data (OHLCV DataFrame)
            config: Configuration object (uses defaults if None)
            checkpoint_dir: Directory for saving model checkpoints
        """
        self.data = data
        self.config = config if config is not None else Config()
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create environment
        self.env = BinaryOptionsEnvironment(
            data=data,
            lookback_window=self.config.training.get('lookback_window', 100),
            trade_duration=60,
            initial_balance=10000,
            trade_amount=100,
            payout_ratio=self.config.trading.get('payout_ratio', 0.8)
        )
        
        # Create agent
        self.agent = DQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_space,
            learning_rate=self.config.training.get('learning_rate', 0.0001),
            gamma=self.config.training.get('gamma', 0.99),
            epsilon_start=self.config.training.get('epsilon_start', 1.0),
            epsilon_end=self.config.training.get('epsilon_end', 0.01),
            epsilon_decay=self.config.training.get('epsilon_decay', 0.995),
            batch_size=self.config.training.get('batch_size', 512),
            memory_size=self.config.training.get('memory_size', 100000),
            hidden_size=4096,
            use_advanced_model=True
        )
        
        # Training history
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_win_rates = []
        self.episode_balances = []
        
        print("[Trainer] Initialized successfully")
        print(f"[Data] Training samples: {len(data)}")
        print(f"[Environment] State size: {self.env.state_size}")
        print(f"[Environment] Action size: {self.env.action_space}")
    
    def train(
        self,
        episodes: int = 1000,
        save_frequency: int = 100,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train the agent
        
        Args:
            episodes: Number of episodes to train
            save_frequency: How often to save checkpoints
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing training metrics
        """
        print(f"\n[Training] Starting training for {episodes} episodes")
        print(f"[Training] Checkpoint frequency: every {save_frequency} episodes")
        
        for episode in tqdm(range(episodes), desc="Training"):
            state = self.env.reset()
            episode_reward = 0
            episode_losses = []
            done = False
            
            while not done:
                # Select and perform action
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train the agent
                loss = self.agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
                
                episode_reward += reward
                state = next_state
            
            # Record metrics
            stats = self.env.get_stats()
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(np.mean(episode_losses) if episode_losses else 0)
            self.episode_win_rates.append(stats['win_rate'])
            self.episode_balances.append(stats['final_balance'])
            
            # Update learning rate scheduler
            self.agent.scheduler.step(stats['win_rate'])
            
            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_loss = np.mean(self.episode_losses[-10:])
                avg_win_rate = np.mean(self.episode_win_rates[-10:])
                print(f"\n[Episode {episode + 1}/{episodes}]")
                print(f"  Reward: {episode_reward:.2f} (avg: {avg_reward:.2f})")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Win Rate: {stats['win_rate']:.2%} (avg: {avg_win_rate:.2%})")
                print(f"  Balance: ${stats['final_balance']:.2f}")
                print(f"  Epsilon: {self.agent.epsilon:.4f}")
                print(f"  Trades: {stats['total_trades']}")
            
            # Save checkpoint
            if (episode + 1) % save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"winston_ai_episode_{episode + 1}.pth"
                )
                self.agent.save(
                    checkpoint_path,
                    episode=episode + 1,
                    metrics={
                        'reward': episode_reward,
                        'win_rate': stats['win_rate'],
                        'balance': stats['final_balance']
                    }
                )
        
        print("\n[Training] Training completed!")
        
        # Save final model
        final_path = os.path.join(self.checkpoint_dir, "winston_ai_final.pth")
        self.agent.save(final_path, episode=episodes)
        
        return {
            'rewards': self.episode_rewards,
            'losses': self.episode_losses,
            'win_rates': self.episode_win_rates,
            'balances': self.episode_balances
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot training results
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        axes[0, 0].plot(self._moving_average(self.episode_rewards, 50), label='MA(50)', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot losses
        axes[0, 1].plot(self.episode_losses, alpha=0.6, label='Episode Loss')
        axes[0, 1].plot(self._moving_average(self.episode_losses, 50), label='MA(50)', linewidth=2)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot win rates
        axes[1, 0].plot(self.episode_win_rates, alpha=0.6, label='Win Rate')
        axes[1, 0].plot(self._moving_average(self.episode_win_rates, 50), label='MA(50)', linewidth=2)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% baseline')
        axes[1, 0].set_title('Win Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot balances
        axes[1, 1].plot(self.episode_balances, alpha=0.6, label='Final Balance')
        axes[1, 1].plot(self._moving_average(self.episode_balances, 50), label='MA(50)', linewidth=2)
        axes[1, 1].axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Initial balance')
        axes[1, 1].set_title('Account Balance')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Balance ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Trainer] Plot saved to {save_path}")
        else:
            plt.show()
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint (finds latest if None)
        """
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(self.checkpoint_dir)
            if checkpoint_path is None:
                print("[Trainer] No checkpoints found")
                return
        
        self.agent.load(checkpoint_path)
        print(f"[Trainer] Loaded checkpoint from {checkpoint_path}")
    
    @staticmethod
    def _moving_average(data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window:
            return data
        return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
