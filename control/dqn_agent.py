"""
DQN Agent for Humanoid Walking
Complete implementation with Q-Network, Replay Buffer, and Training Logic
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json


# Experience tuple
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """
    Q-Network for DQN.
    
    Architecture:
        Input: state_dim (29)
        Hidden: fc1(256) -> fc2(256) -> fc3(128)
        Output: num_actions (40 for 8 joints × 5 bins)
    
    The output is reshaped to (8, 5) for per-joint action selection.
    """
    
    def __init__(self, 
                 state_dim: int,
                 num_joints: int = 8,
                 num_bins: int = 5,
                 hidden_dims: List[int] = [256, 256, 128]):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state space
            num_joints: Number of controllable joints
            num_bins: Number of discrete torque values per joint
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.num_joints = num_joints
        self.num_bins = num_bins
        self.num_actions = num_joints * num_bins
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Prevent overfitting
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            Q-values (batch_size, num_joints, num_bins)
        """
        # Forward through network
        q_values = self.network(state)
        
        # Reshape to (batch_size, num_joints, num_bins)
        batch_size = state.shape[0]
        q_values = q_values.view(batch_size, self.num_joints, self.num_bins)
        
        return q_values
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State tensor (state_dim,) or (1, state_dim)
            epsilon: Exploration rate
        
        Returns:
            Action array (num_joints,) with discrete bin indices
        """
        if random.random() < epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.num_bins, size=self.num_joints)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                
                q_values = self.forward(state)  # (1, num_joints, num_bins)
                
                # Select action with highest Q-value for each joint
                actions = q_values.argmax(dim=2).squeeze(0).cpu().numpy()
                
                return actions


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    Stores transitions and samples random mini-batches.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for Humanoid Walking.
    
    Features:
        - Double DQN (reduces overestimation)
        - Target network (improves stability)
        - Experience replay
        - Epsilon-greedy exploration
        - Gradient clipping
    """
    
    def __init__(self,
                 state_dim: int,
                 num_joints: int = 8,
                 num_bins: int = 5,
                 hidden_dims: List[int] = [256, 256, 128],
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 100000,
                 batch_size: int = 128,
                 target_update_freq: int = 1000,
                 device: Optional[str] = None):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of state space (29 for humanoid)
            num_joints: Number of joints (8)
            num_bins: Discrete torque bins per joint (5)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Replay buffer size
            batch_size: Mini-batch size for training
            target_update_freq: Steps between target network updates
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"DQN Agent using device: {self.device}")
        
        self.state_dim = state_dim
        self.num_joints = num_joints
        self.num_bins = num_bins
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.policy_net = QNetwork(
            state_dim, num_joints, num_bins, hidden_dims
        ).to(self.device)
        
        self.target_net = QNetwork(
            state_dim, num_joints, num_bins, hidden_dims
        ).to(self.device)
        
        # Copy weights from policy to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.steps_done = 0
        self.episodes_done = 0
        self.training_losses = []
        
        print(f"DQN Agent initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action space: {num_joints} joints × {num_bins} bins = {num_joints * num_bins} actions")
        print(f"  Network: {hidden_dims}")
        print(f"  Replay buffer: {buffer_capacity}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (numpy array)
            training: Whether in training mode (uses epsilon)
        
        Returns:
            Action array (num_joints,)
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        if training:
            epsilon = self.epsilon
        else:
            epsilon = 0.0  # Greedy policy during evaluation
        
        action = self.policy_net.select_action(state_tensor, epsilon)
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (if enough experiences).
        
        Returns:
            Loss value if training happened, None otherwise
        """
        # Need enough experiences to sample a batch
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        # policy_net output: (batch_size, num_joints, num_bins)
        current_q_values = self.policy_net(states)
        
        # Gather Q-values for actions taken
        # actions shape: (batch_size, num_joints)
        # Need to gather along the bins dimension
        actions_expanded = actions.unsqueeze(2)  # (batch_size, num_joints, 1)
        current_q = current_q_values.gather(2, actions_expanded).squeeze(2)  # (batch_size, num_joints)
        current_q = current_q.mean(dim=1)  # Average over joints -> (batch_size,)
        
        # Compute next Q-values using Double DQN
        with torch.no_grad():
            # Use policy network to select actions
            next_q_values_policy = self.policy_net(next_states)
            next_actions = next_q_values_policy.argmax(dim=2)  # (batch_size, num_joints)
            
            # Use target network to evaluate actions
            next_q_values_target = self.target_net(next_states)
            next_actions_expanded = next_actions.unsqueeze(2)
            next_q = next_q_values_target.gather(2, next_actions_expanded).squeeze(2)
            next_q = next_q.mean(dim=1)  # (batch_size,)
            
            # Compute target Q-values
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self, filepath: str, episode: int, total_reward: float):
        """
        Save agent checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            episode: Current episode number
            total_reward: Total reward achieved
        """
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'total_reward': total_reward,
            'training_losses': self.training_losses[-100:],  # Last 100 losses
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load agent checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episode']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"  Episode: {self.episodes_done}")
        print(f"  Steps: {self.steps_done}")
        print(f"  Epsilon: {self.epsilon:.4f}")
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'episodes': self.episodes_done,
            'steps': self.steps_done,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0
        }


# Utility function for evaluation
def evaluate_agent(agent: DQNAgent, 
                   env,
                   num_episodes: int = 10,
                   render: bool = False) -> Dict:
    """
    Evaluate agent performance.
    
    Args:
        agent: DQN agent to evaluate
        env: Environment
        num_episodes: Number of episodes to run
        render: Whether to render episodes
    
    Returns:
        Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        initial_x = info['torso_x']
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (greedy, no exploration)
            action = agent.select_action(state, training=False)
            
            # Step environment
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if render:
                env.render()
        
        final_x = info['torso_x']
        distance = final_x - initial_x
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_distances.append(distance)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_distance': np.mean(episode_distances),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards)
    }


if __name__ == "__main__":
    print("Testing DQN Agent...")
    
    # Test Q-Network
    print("\n1. Testing Q-Network")
    state_dim = 29
    net = QNetwork(state_dim, num_joints=8, num_bins=5)
    
    # Test forward pass
    test_state = torch.randn(4, state_dim)  # Batch of 4
    q_values = net(test_state)
    print(f"  Input shape: {test_state.shape}")
    print(f"  Q-values shape: {q_values.shape}")  # Should be (4, 8, 5)
    
    # Test action selection
    action = net.select_action(test_state[0], epsilon=0.0)
    print(f"  Selected action: {action}")
    
    # Test Replay Buffer
    print("\n2. Testing Replay Buffer")
    buffer = ReplayBuffer(capacity=1000)
    
    for i in range(10):
        buffer.push(
            state=np.random.randn(state_dim),
            action=np.random.randint(0, 5, size=8),
            reward=np.random.randn(),
            next_state=np.random.randn(state_dim),
            done=False
        )
    
    print(f"  Buffer size: {len(buffer)}")
    
    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(5)
    print(f"  Sampled batch shapes:")
    print(f"    States: {states.shape}")
    print(f"    Actions: {actions.shape}")
    print(f"    Rewards: {rewards.shape}")
    
    # Test DQN Agent
    print("\n3. Testing DQN Agent")
    agent = DQNAgent(
        state_dim=state_dim,
        num_joints=8,
        num_bins=5,
        batch_size=4
    )
    
    # Add some experiences
    for i in range(10):
        agent.store_experience(
            state=np.random.randn(state_dim),
            action=np.random.randint(0, 5, size=8),
            reward=np.random.randn(),
            next_state=np.random.randn(state_dim),
            done=False
        )
    
    # Try training step
    loss = agent.train_step()
    print(f"  Training loss: {loss}")
    
    # Test action selection
    test_state = np.random.randn(state_dim)
    action = agent.select_action(test_state)
    print(f"  Selected action: {action}")
    
    print("\n✓ All DQN components working!")