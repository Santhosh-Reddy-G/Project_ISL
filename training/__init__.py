"""
Training Module - Complete Training Pipeline
Orchestrates DQN training with pose library integration
"""

from .trainer import (
    Trainer,
    train_agent,
    evaluate_trained_agent
)

__all__ = [
    'Trainer',
    'train_agent',
    'evaluate_trained_agent'
]

__version__ = '1.0.0'