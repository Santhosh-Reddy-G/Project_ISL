"""
Control Module - DQN Agent for Humanoid Locomotion
Implements Deep Q-Network with experience replay for walking control
"""

from .dqn_agent import (
    QNetwork,
    ReplayBuffer,
    DQNAgent
)

__all__ = [
    'QNetwork',
    'ReplayBuffer',
    'DQNAgent'
]

__version__ = '1.0.0'