"""
Deep Q-Network (DQN) agent for reinforcement learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple, List

from winston_ai.models.winston_model import WinstonAI, AdvancedWinstonAI
from winston_ai.utils.device import get_device


class DQNAgent:
    """
    Deep Q-Network agent for training WinstonAI models
    
    This agent implements:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Soft updates
    - Mixed precision training
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 512,
        memory_size: int = 100000,
        hidden_size: int = 4096,
        use_advanced_model: bool = True
    ):
        """
        Initialize DQN agent
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            batch_size: Batch size for training
            memory_size: Size of replay buffer
            hidden_size: Size of hidden layers in neural network
            use_advanced_model: Whether to use AdvancedWinstonAI model
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = 0.005  # Soft update parameter
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Get device
        self.device = get_device()
        
        # Create neural networks
        model_class = AdvancedWinstonAI if use_advanced_model else WinstonAI
        self.q_network = model_class(state_size, action_size, hidden_size).to(self.device)
        self.target_network = model_class(state_size, action_size, hidden_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=50
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Copy weights to target network
        self.update_target_network()
        
        # Training statistics
        self.step_count = 0
        self.episode_count = 0
        
        print(f"[Agent] DQN Agent initialized")
        print(f"[Model] Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"[Memory] Buffer Size: {memory_size:,}")
        print(f"[Training] Batch Size: {batch_size}")
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action
        """
        if training and np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def replay(self) -> float:
        """
        Train the model on a batch of experiences
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        self.optimizer.zero_grad()
        
        with autocast():
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network (Double DQN)
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute loss (Huber loss for stability)
            loss = F.huber_loss(current_q_values.squeeze(), target_q_values)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Soft update target network
        self.soft_update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def soft_update_target_network(self):
        """Soft update of target network parameters"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath: str, episode: Optional[int] = None, metrics: Optional[dict] = None):
        """
        Save agent state
        
        Args:
            filepath: Path to save checkpoint
            episode: Current episode number
            metrics: Training metrics to save
        """
        save_dict = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
        }
        
        if episode is not None:
            save_dict['episode'] = episode
        
        if metrics is not None:
            save_dict['metrics'] = metrics
        
        torch.save(save_dict, filepath)
        print(f"[Agent] Saved checkpoint to {filepath}")
    
    def load(self, filepath: str, load_optimizer: bool = True):
        """
        Load agent state
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        
        if 'step_count' in checkpoint:
            self.step_count = checkpoint['step_count']
        
        if 'episode_count' in checkpoint:
            self.episode_count = checkpoint['episode_count']
        
        print(f"[Agent] Loaded checkpoint from {filepath}")
        
        return checkpoint
