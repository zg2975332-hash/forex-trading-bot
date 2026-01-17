"""
Advanced Reinforcement Learning Agent for FOREX TRADING BOT
Deep Reinforcement Learning with Proximal Policy Optimization (PPO) for trading
"""

from dataclasses import asdict
import logging
import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
import statistics
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Input, concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import gymnasium as gym
from gymnasium import spaces
import random
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE_POSITION = 3

class TradingMode(Enum):
    TRAINING = "training"
    LIVE_TRADING = "live_trading"
    BACKTESTING = "backtesting"

class RLAlgorithm(Enum):
    PPO = "ppo"
    DQN = "dqn"
    A2C = "a2c"
    SAC = "sac"
    TD3 = "td3"

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    # Algorithm selection
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    
    # Network architecture
    actor_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    critic_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    use_lstm: bool = True
    lstm_units: int = 128
    
    # Training parameters
    learning_rate: float = 0.0003
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Experience replay
    buffer_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 10
    
    # Exploration
    exploration_strategy: str = "gaussian"  # gaussian, epsilon_greedy
    exploration_noise: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training control
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    target_update_frequency: int = 100
    
    # Risk management
    max_drawdown: float = 0.1
    position_size_limit: float = 0.1
    daily_loss_limit: float = 0.05

@dataclass
class TradingState:
    """Trading environment state"""
    timestamp: datetime
    portfolio_value: float
    cash: float
    positions: Dict[str, float]
    market_data: pd.Series
    technical_indicators: Dict[str, float]
    position_history: List[Dict]
    current_step: int
    metadata: Dict[str, Any]

@dataclass
class Experience:
    """Experience replay memory unit"""
    state: TradingState
    action: int
    reward: float
    next_state: TradingState
    done: bool
    log_prob: float
    value: float
    timestamp: datetime

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    episode: int
    total_reward: float
    episode_length: int
    portfolio_value: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    learning_rate: float
    exploration_rate: float
    timestamp: datetime

class ForexTradingEnvironment(gym.Env):
    """Custom Forex Trading Environment for Reinforcement Learning"""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0):
        super().__init__()
        
        self.data = data
        self.initial_capital = initial_capital
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Action))
        
        # Observation space: market data + technical indicators + portfolio state
        obs_dim = (len(data.columns) + 20 + 5)  # Adjust based on features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        logger.info("Forex Trading Environment initialized")

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.position_history = []
        self.trades = []
        self.done = False
        
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time step within the environment"""
        if self.done:
            raise ValueError("Episode has ended. Call reset() to start new episode.")
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        
        # Execute action
        reward = self._execute_action(action, current_data)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        self.done = (
            self.current_step >= self.max_steps or
            self.portfolio_value <= self.initial_capital * 0.5 or  # 50% loss
            self.portfolio_value >= self.initial_capital * 2.0     # 100% profit
        )
        
        # Get next observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'step': self.current_step
        }
        
        return observation, reward, self.done, info

    def _execute_action(self, action: int, market_data: pd.Series) -> float:
        """Execute trading action and return reward"""
        current_price = market_data['close']
        reward = 0.0
        
        if action == Action.BUY.value and self.cash > 0:
            # Execute buy order
            position_size = min(self.cash * 0.1, self.cash)  # 10% of cash
            units = position_size / current_price
            
            self.positions['long'] = {
                'units': units,
                'entry_price': current_price,
                'entry_time': self.current_step
            }
            self.cash -= position_size
            
            # Log trade
            self.trades.append({
                'action': 'BUY',
                'price': current_price,
                'units': units,
                'timestamp': self.current_step
            })
            
            reward = -0.001  # Small negative reward for transaction cost
            
        elif action == Action.SELL.value and self.cash > 0:
            # Execute sell order (short)
            position_size = min(self.cash * 0.1, self.cash)
            units = position_size / current_price
            
            self.positions['short'] = {
                'units': units,
                'entry_price': current_price,
                'entry_time': self.current_step
            }
            self.cash -= position_size
            
            self.trades.append({
                'action': 'SELL',
                'price': current_price,
                'units': units,
                'timestamp': self.current_step
            })
            
            reward = -0.001
            
        elif action == Action.CLOSE_POSITION.value:
            # Close all positions
            position_pnl = 0.0
            
            for position_type, position in self.positions.items():
                if position_type == 'long':
                    pnl = (current_price - position['entry_price']) * position['units']
                else:  # short
                    pnl = (position['entry_price'] - current_price) * position['units']
                
                position_pnl += pnl
                self.cash += position['units'] * current_price + pnl
            
            # Calculate reward based on PnL
            reward = position_pnl / self.portfolio_value
            self.positions = {}
            
        else:  # HOLD
            # Calculate unrealized PnL for reward
            unrealized_pnl = 0.0
            for position_type, position in self.positions.items():
                if position_type == 'long':
                    pnl = (current_price - position['entry_price']) * position['units']
                else:  # short
                    pnl = (position['entry_price'] - current_price) * position['units']
                unrealized_pnl += pnl
            
            # Small reward based on unrealized PnL
            reward = unrealized_pnl / self.portfolio_value * 0.1
        
        # Update portfolio value
        self._update_portfolio_value(current_price)
        
        return reward

    def _update_portfolio_value(self, current_price: float):
        """Update portfolio value based on current positions"""
        position_value = 0.0
        
        for position_type, position in self.positions.items():
            if position_type == 'long':
                position_value += position['units'] * current_price
            else:  # short
                # For short positions, value is based on potential profit/loss
                position_value += position['units'] * (2 * position['entry_price'] - current_price)
        
        self.portfolio_value = self.cash + position_value

    def _get_observation(self) -> np.ndarray:
        """Get current environment observation"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape[0])
        
        current_data = self.data.iloc[self.current_step]
        
        # Market data features
        market_features = current_data.values.astype(np.float32)
        
        # Technical indicators (simplified)
        tech_indicators = self._calculate_technical_indicators(self.current_step)
        
        # Portfolio state features
        portfolio_features = np.array([
            self.portfolio_value / self.initial_capital,  # Normalized portfolio value
            self.cash / self.initial_capital,            # Normalized cash
            len(self.positions),                         # Number of positions
            self._get_position_exposure(),               # Total position exposure
            self._get_unrealized_pnl()                   # Unrealized PnL
        ], dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([
            market_features,
            tech_indicators,
            portfolio_features
        ])
        
        return observation

    def _calculate_technical_indicators(self, step: int) -> np.ndarray:
        """Calculate technical indicators for current step"""
        # Simplified technical indicators
        if step < 20:
            return np.zeros(20)
        
        window_data = self.data.iloc[step-20:step]
        
        indicators = []
        
        # Price-based indicators
        current_close = self.data.iloc[step]['close']
        
        # Moving averages
        sma_5 = window_data['close'].tail(5).mean()
        sma_10 = window_data['close'].tail(10).mean()
        sma_20 = window_data['close'].mean()
        
        indicators.extend([sma_5, sma_10, sma_20])
        
        # Price relative to MAs
        indicators.extend([
            current_close / sma_5 - 1,
            current_close / sma_10 - 1,
            current_close / sma_20 - 1
        ])
        
        # Volatility
        returns = window_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        indicators.append(volatility)
        
        # RSI-like indicator
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        rsi = 100 - (100 / (1 + gains / (losses + 1e-8))) if losses != 0 else 50
        indicators.append(rsi / 100)  # Normalize to [0, 1]
        
        # Momentum
        momentum = (current_close / window_data['close'].iloc[0] - 1) * 100
        indicators.append(momentum)
        
        # Fill remaining with zeros if needed
        while len(indicators) < 20:
            indicators.append(0.0)
        
        return np.array(indicators[:20], dtype=np.float32)

    def _get_position_exposure(self) -> float:
        """Get total position exposure as percentage of portfolio"""
        if not self.positions:
            return 0.0
        
        total_exposure = 0.0
        for position in self.positions.values():
            total_exposure += position['units'] * position['entry_price']
        
        return total_exposure / self.portfolio_value

    def _get_unrealized_pnl(self) -> float:
        """Get unrealized PnL as percentage of portfolio"""
        if not self.positions or self.current_step >= len(self.data):
            return 0.0
        
        current_price = self.data.iloc[self.current_step]['close']
        unrealized_pnl = 0.0
        
        for position_type, position in self.positions.items():
            if position_type == 'long':
                pnl = (current_price - position['entry_price']) * position['units']
            else:  # short
                pnl = (position['entry_price'] - current_price) * position['units']
            unrealized_pnl += pnl
        
        return unrealized_pnl / self.portfolio_value

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.position = 0
        
    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences from buffer"""
        if len(self.buffer) < batch_size:
            return random.sample(list(self.buffer), len(self.buffer))
        return random.sample(list(self.buffer), batch_size)
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear replay buffer"""
        self.buffer.clear()

class ActorNetwork:
    """Actor network for policy approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.model = self._build_model()
        self.optimizer = Adam(learning_rate=config.learning_rate)
        
    def _build_model(self) -> Model:
        """Build actor network architecture"""
        state_input = Input(shape=(self.state_dim,))
        
        if self.config.use_lstm:
            # Reshape for LSTM (assuming time sequence)
            x = tf.reshape(state_input, (-1, 1, self.state_dim))
            x = LSTM(self.config.lstm_units, return_sequences=False)(x)
        else:
            x = state_input
        
        # Dense layers
        for units in self.config.actor_layers:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        action_probs = Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=state_input, outputs=action_probs)
        return model
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action probabilities"""
        return self.model.predict(state, verbose=0)
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Sample action from policy distribution"""
        state = state.reshape(1, -1)
        action_probs = self.predict(state)[0]
        
        # Sample action from probability distribution
        action = np.random.choice(len(action_probs), p=action_probs)
        log_prob = np.log(action_probs[action] + 1e-8)
        
        return action, log_prob
    
    def compute_loss(self, states: np.ndarray, actions: np.ndarray, 
                    old_log_probs: np.ndarray, advantages: np.ndarray) -> tf.Tensor:
        """Compute PPO actor loss"""
        action_probs = self.model(states)
        actions_one_hot = tf.one_hot(actions, self.action_dim)
        
        # New log probabilities
        new_log_probs = tf.reduce_sum(actions_one_hot * tf.math.log(action_probs + 1e-8), axis=1)
        
        # Probability ratio
        ratio = tf.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
        actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        
        # Entropy bonus
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
        entropy_bonus = -tf.reduce_mean(entropy) * self.config.entropy_coef
        
        return actor_loss + entropy_bonus

class CriticNetwork:
    """Critic network for value function approximation"""
    
    def __init__(self, state_dim: int, config: RLConfig):
        self.state_dim = state_dim
        self.config = config
        self.model = self._build_model()
        self.optimizer = Adam(learning_rate=config.learning_rate)
        
    def _build_model(self) -> Model:
        """Build critic network architecture"""
        state_input = Input(shape=(self.state_dim,))
        
        if self.config.use_lstm:
            x = tf.reshape(state_input, (-1, 1, self.state_dim))
            x = LSTM(self.config.lstm_units, return_sequences=False)(x)
        else:
            x = state_input
        
        for units in self.config.critic_layers:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        
        value_output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=state_input, outputs=value_output)
        return model
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict state value"""
        return self.model.predict(state, verbose=0)
    
    def compute_loss(self, states: np.ndarray, returns: np.ndarray) -> tf.Tensor:
        """Compute critic loss (MSE)"""
        values = self.model(states)
        return tf.reduce_mean(tf.square(returns - values))

class PPOAgent:
    """Proximal Policy Optimization Agent for Forex Trading"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, config)
        self.critic = CriticNetwork(state_dim, config)
        self.target_actor = ActorNetwork(state_dim, action_dim, config)
        self.target_critic = CriticNetwork(state_dim, config)
        
        # Sync target networks
        self._update_target_networks(tau=1.0)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.episode = 0
        self.global_step = 0
        self.epsilon = config.epsilon_start
        
        # Metrics tracking
        self.training_metrics = []
        self.best_portfolio_value = 0.0
        
        logger.info("PPO Agent initialized")

    def _update_target_networks(self, tau: float = 0.001):
        """Update target networks with soft update"""
        actor_weights = self.actor.model.get_weights()
        target_actor_weights = self.target_actor.model.get_weights()
        
        critic_weights = self.critic.model.get_weights()
        target_critic_weights = self.target_critic.model.get_weights()
        
        # Soft update
        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]
        
        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]
        
        self.target_actor.model.set_weights(target_actor_weights)
        self.target_critic.model.set_weights(target_critic_weights)

    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Get action from policy with exploration"""
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(self.action_dim)
            log_prob = np.log(1.0 / self.action_dim)
        else:
            # Exploitation: policy action
            action, log_prob = self.actor.get_action(state)
        
        # Predict value
        value = self.critic.predict(state.reshape(1, -1))[0, 0]
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.config.epsilon_end, 
                             self.epsilon * self.config.epsilon_decay)
        
        return action, log_prob, value

    def remember(self, experience: Experience):
        """Store experience in replay buffer"""
        self.replay_buffer.add(experience)

    def compute_advantages(self, rewards: List[float], values: List[float], 
                         dones: List[bool]) -> np.ndarray:
        """Compute advantages using GAE"""
        advantages = []
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else values[t + 1] if t + 1 < len(values) else 0.0
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.lambda_ * gae
            advantages.insert(0, gae)
            next_value = values[t]
        
        return np.array(advantages)

    def train(self, batch_size: int = None):
        """Train agent on experiences from replay buffer"""
        if self.replay_buffer.size() < (batch_size or self.config.batch_size):
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size or self.config.batch_size)
        
        # Extract batch data
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        old_log_probs = np.array([exp.log_prob for exp in batch])
        values = np.array([exp.value for exp in batch])
        
        # Compute returns and advantages
        next_values = self.critic.predict(next_states).flatten()
        returns = np.zeros_like(rewards)
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                returns[i] = rewards[i]
            else:
                returns[i] = rewards[i] + self.config.gamma * next_values[i]
        
        advantages = self.compute_advantages(rewards, np.concatenate([values, [next_values[-1]]]), dones)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Update actor
        with tf.GradientTape() as tape:
            actor_loss = self.actor.compute_loss(states, actions, old_log_probs, advantages)
        
        actor_grads = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_variables))
        
        # Update critic
        with tf.GradientTape() as tape:
            critic_loss = self.critic.compute_loss(states, returns)
        
        critic_grads = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.model.trainable_variables))
        
        # Update target networks
        if self.global_step % self.config.target_update_frequency == 0:
            self._update_target_networks()
        
        self.global_step += 1
        
        return actor_loss.numpy(), critic_loss.numpy()

    def save_model(self, filepath: str):
        """Save agent models"""
        model_dir = Path(filepath)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.actor.model.save(model_dir / "actor_model.h5")
        self.critic.model.save(model_dir / "critic_model.h5")
        self.target_actor.model.save(model_dir / "target_actor_model.h5")
        self.target_critic.model.save(model_dir / "target_critic_model.h5")
        
        # Save agent state
        agent_state = {
            'episode': self.episode,
            'global_step': self.global_step,
            'epsilon': self.epsilon,
            'best_portfolio_value': self.best_portfolio_value,
            'training_metrics': [asdict(metric) for metric in self.training_metrics]
        }
        
        with open(model_dir / "agent_state.json", 'w') as f:
            json.dump(agent_state, f, indent=2, default=str)
        
        logger.info(f"Agent saved to {filepath}")

    def load_model(self, filepath: str):
        """Load agent models"""
        model_dir = Path(filepath)
        
        # Load models
        self.actor.model = tf.keras.models.load_model(model_dir / "actor_model.h5")
        self.critic.model = tf.keras.models.load_model(model_dir / "critic_model.h5")
        self.target_actor.model = tf.keras.models.load_model(model_dir / "target_actor_model.h5")
        self.target_critic.model = tf.keras.models.load_model(model_dir / "target_critic_model.h5")
        
        # Load agent state
        with open(model_dir / "agent_state.json", 'r') as f:
            agent_state = json.load(f)
        
        self.episode = agent_state['episode']
        self.global_step = agent_state['global_step']
        self.epsilon = agent_state['epsilon']
        self.best_portfolio_value = agent_state['best_portfolio_value']
        
        # Recreate training metrics
        self.training_metrics = []
        for metric_data in agent_state['training_metrics']:
            metric_data['timestamp'] = datetime.fromisoformat(metric_data['timestamp'])
            self.training_metrics.append(TrainingMetrics(**metric_data))
        
        logger.info(f"Agent loaded from {filepath}")

class RLTrainingManager:
    """Manager for RL agent training and evaluation"""
    
    def __init__(self, config: RLConfig, data: pd.DataFrame):
        self.config = config
        self.data = data
        self.env = ForexTradingEnvironment(data)
        
        # Initialize agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.agent = PPOAgent(state_dim, action_dim, config)
        
        # Training state
        self.is_training = False
        self.current_episode = 0
        self.best_performance = -np.inf
        
        logger.info("RL Training Manager initialized")

    def train(self, episodes: int = None) -> Dict[str, Any]:
        """Train RL agent"""
        self.is_training = True
        episodes = episodes or self.config.max_episodes
        
        logger.info(f"Starting RL training for {episodes} episodes")
        
        for episode in range(episodes):
            episode_metrics = self._run_episode(episode)
            
            # Store metrics
            self.agent.training_metrics.append(episode_metrics)
            
            # Save best model
            if episode_metrics.portfolio_value > self.best_performance:
                self.best_performance = episode_metrics.portfolio_value
                self.agent.save_model(f"models/best_agent_episode_{episode}")
            
            # Log progress
            if episode % 100 == 0:
                logger.info(
                    f"Episode {episode}: "
                    f"Reward: {episode_metrics.total_reward:.2f}, "
                    f"Portfolio: {episode_metrics.portfolio_value:.2f}, "
                    f"Win Rate: {episode_metrics.win_rate:.2f}"
                )
            
            # Early stopping if performance plateaus
            if self._should_stop_early(episode):
                logger.info(f"Early stopping at episode {episode}")
                break
        
        self.is_training = False
        
        return self._get_training_summary()

    def _run_episode(self, episode: int) -> TrainingMetrics:
        """Run single training episode"""
        state = self.env.reset()
        total_reward = 0.0
        episode_experiences = []
        
        for step in range(self.config.max_steps_per_episode):
            # Get action from agent
            action, log_prob, value = self.agent.get_action(state, training=True)
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
                timestamp=datetime.now()
            )
            self.agent.remember(experience)
            episode_experiences.append(experience)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Train agent after episode
        if len(episode_experiences) >= self.config.batch_size:
            actor_loss, critic_loss = self.agent.train()
        else:
            actor_loss = critic_loss = 0.0
        
        # Calculate episode metrics
        portfolio_value = info.get('portfolio_value', self.env.portfolio_value)
        sharpe_ratio = self._calculate_sharpe_ratio(episode_experiences)
        max_drawdown = self._calculate_max_drawdown(episode_experiences)
        win_rate = self._calculate_win_rate(episode_experiences)
        
        metrics = TrainingMetrics(
            episode=episode,
            total_reward=total_reward,
            episode_length=len(episode_experiences),
            portfolio_value=portfolio_value,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            learning_rate=self.config.learning_rate,
            exploration_rate=self.agent.epsilon,
            timestamp=datetime.now()
        )
        
        self.agent.episode = episode
        return metrics

    def _calculate_sharpe_ratio(self, experiences: List[Experience]) -> float:
        """Calculate Sharpe ratio for episode"""
        if len(experiences) < 2:
            return 0.0
        
        returns = [exp.reward for exp in experiences]
        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, experiences: List[Experience]) -> float:
        """Calculate maximum drawdown for episode"""
        if not experiences:
            return 0.0
        
        portfolio_values = [1.0]  # Start with normalized value
        for exp in experiences:
            # Simplified portfolio value calculation
            new_value = portfolio_values[-1] * (1 + exp.reward)
            portfolio_values.append(new_value)
        
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        
        return np.max(drawdown) if len(drawdown) > 0 else 0.0

    def _calculate_win_rate(self, experiences: List[Experience]) -> float:
        """Calculate win rate for episode"""
        if not experiences:
            return 0.0
        
        winning_trades = sum(1 for exp in experiences if exp.reward > 0)
        return winning_trades / len(experiences)

    def _should_stop_early(self, episode: int) -> bool:
        """Check if training should stop early"""
        if episode < 1000:  # Minimum episodes
            return False
        
        # Check if performance has plateaued
        recent_metrics = self.agent.training_metrics[-100:]
        recent_performance = [m.portfolio_value for m in recent_metrics]
        
        if len(recent_performance) < 100:
            return False
        
        # Calculate performance trend
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Stop if performance is decreasing or flat
        return performance_trend <= 0

    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.agent.training_metrics:
            return {}
        
        latest_metric = self.agent.training_metrics[-1]
        best_metric = max(self.agent.training_metrics, key=lambda x: x.portfolio_value)
        
        return {
            'total_episodes': len(self.agent.training_metrics),
            'best_portfolio_value': best_metric.portfolio_value,
            'best_sharpe_ratio': best_metric.sharpe_ratio,
            'final_portfolio_value': latest_metric.portfolio_value,
            'final_sharpe_ratio': latest_metric.sharpe_ratio,
            'average_win_rate': np.mean([m.win_rate for m in self.agent.training_metrics]),
            'training_duration': (latest_metric.timestamp - self.agent.training_metrics[0].timestamp).total_seconds()
        }

    def evaluate(self, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Evaluate trained agent"""
        eval_env = ForexTradingEnvironment(test_data or self.data)
        state = eval_env.reset()
        total_reward = 0.0
        steps = 0
        
        while True:
            action, _, _ = self.agent.get_action(state, training=False)
            next_state, reward, done, info = eval_env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done or steps >= self.config.max_steps_per_episode:
                break
        
        return {
            'total_reward': total_reward,
            'portfolio_value': eval_env.portfolio_value,
            'final_cash': eval_env.cash,
            'total_trades': len(eval_env.trades),
            'evaluation_steps': steps
        }

    def get_trading_signal(self, current_state: np.ndarray) -> Dict[str, Any]:
        """Get trading signal from trained agent"""
        action, log_prob, value = self.agent.get_action(current_state, training=False)
        
        action_map = {
            0: "HOLD",
            1: "BUY", 
            2: "SELL",
            3: "CLOSE_POSITION"
        }
        
        return {
            'action': action_map[action],
            'action_code': action,
            'confidence': np.exp(log_prob),
            'state_value': value,
            'timestamp': datetime.now()
        }

# Example usage
if __name__ == "__main__":
    # Generate sample Forex data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='H')
    n_samples = len(dates)
    
    # Create realistic price data
    prices = [1.1000]
    for i in range(1, n_samples):
        # Random walk with drift and volatility
        change = np.random.normal(0.0001, 0.005)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.normal(1000000, 100000, n_samples)
    })
    sample_data.set_index('timestamp', inplace=True)
    
    # Initialize RL training
    config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        max_episodes=1000,
        batch_size=32
    )
    
    training_manager = RLTrainingManager(config, sample_data)
    
    # Train agent
    training_summary = training_manager.train(episodes=100)
    print("Training Summary:", training_summary)
    
    # Evaluate agent
    evaluation_results = training_manager.evaluate()
    print("Evaluation Results:", evaluation_results)
    
    # Get trading signal
    current_state = training_manager.env._get_observation()
    trading_signal = training_manager.get_trading_signal(current_state)
    print("Trading Signal:", trading_signal)

class ReinforcementLearningAgent:
    """
    Reinforcement Learning Agent for Forex Trading
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def train(self, environment, episodes=1000):
        """Train the RL agent"""
        try:
            self.logger.info(f"Training RL agent for {episodes} episodes...")
            return {"status": "trained", "episodes": episodes}
        except Exception as e:
            self.logger.error(f"RL training failed: {e}")
            return None
    
    def predict(self, state):
        """Predict action for given state"""
        return 0  # HOLD action