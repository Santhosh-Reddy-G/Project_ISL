"""
Complete Training Pipeline for Humanoid Walking DQN
Integrates all modules: Perception, Simulation, and Control
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List
import time
import json
from tqdm import tqdm

# Import from other modules
from control.dqn_agent import DQNAgent, evaluate_agent
from simulation.humanoid_env import (
    HumanoidWalkEnvDiscrete,
    load_pose_library,
    sample_random_pose
)


class Trainer:
    """
    Complete training pipeline for DQN humanoid walking agent.
    """
    
    def __init__(self,
                 env_config: Optional[Dict] = None,
                 agent_config: Optional[Dict] = None,
                 training_config: Optional[Dict] = None):
        """
        Initialize trainer.
        
        Args:
            env_config: Environment configuration
            agent_config: Agent configuration
            training_config: Training configuration
        """
        # FIXED: Merge configs FIRST, then create directory
        default_training_config = {
            'num_episodes': 5000,
            'eval_freq': 50,
            'eval_episodes': 10,
            'save_freq': 100,
            'log_freq': 10,
            'checkpoint_dir': 'models/checkpoints',
            'use_pose_library': True,
            'pose_library_path': 'data/pose_library',
            'early_stopping': True,
            'patience': 200,
            'min_reward_threshold': 50.0
        }
        
        # Merge user config with defaults
        if training_config:
            default_training_config.update(training_config)
        self.training_config = default_training_config
        
        # Now safe to create directory
        Path(self.training_config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        
        # Default configurations for other modules
        self.env_config = env_config or {
            'render_mode': None,
            'max_steps': 1000,
            'num_torque_bins': 5
        }
        
        self.agent_config = agent_config or {
            'hidden_dims': [256, 256, 128],
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'buffer_capacity': 100000,
            'batch_size': 128,
            'target_update_freq': 1000
        }
        
        # Initialize components
        self.env = None
        self.agent = None
        self.pose_library = None
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.eval_rewards = []
        self.training_losses = []
        
        # Best model tracking
        self.best_eval_reward = -float('inf')
        self.episodes_without_improvement = 0
        
    def setup(self):
        """Setup environment, agent, and pose library."""
        print("="*60)
        print("INITIALIZING TRAINING PIPELINE")
        print("="*60)
        
        # Create environment
        print("\n1. Creating environment...")
        self.env = HumanoidWalkEnvDiscrete(**self.env_config)
        print(f"   ✓ Environment created")
        print(f"     Observation space: {self.env.observation_space.shape}")
        print(f"     Action space: {self.env.action_space}")
        
        # Create agent
        print("\n2. Creating DQN agent...")
        self.agent = DQNAgent(
            state_dim=self.env.observation_space.shape[0],
            num_joints=8,
            num_bins=self.env_config['num_torque_bins'],
            **self.agent_config
        )
        print(f"   ✓ Agent created")
        
        # Load pose library
        if self.training_config['use_pose_library']:
            print("\n3. Loading pose library...")
            pose_path = Path(self.training_config['pose_library_path'])
            
            if pose_path.exists():
                self.pose_library = load_pose_library(str(pose_path))
                print(f"   ✓ Loaded {len(self.pose_library)} poses")
                print(f"     Poses: {list(self.pose_library.keys())[:5]}...")
            else:
                print(f"   ⚠ Pose library not found at {pose_path}")
                print(f"     Training with default standing pose")
                self.pose_library = None
        else:
            print("\n3. Skipping pose library (using default pose)")
            self.pose_library = None
        
        print("\n" + "="*60)
        print("SETUP COMPLETE - Ready to train!")
        print("="*60 + "\n")
    
    def train_episode(self, episode: int) -> Dict:
        """Train for one episode."""
        # Select initial pose
        if self.pose_library:
            initial_pose = sample_random_pose(self.pose_library)
        else:
            initial_pose = None
        
        # Reset environment
        state, info = self.env.reset(initial_pose=initial_pose)
        
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        initial_x = info['torso_x']
        
        done = False
        truncated = False
        
        # Episode loop
        while not (done or truncated):
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            if len(self.agent.memory) >= self.agent.batch_size:
                loss = self.agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                    self.agent.training_losses.append(loss)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # Decay epsilon
        self.agent.decay_epsilon()
        self.agent.episodes_done += 1
        
        # Calculate distance traveled
        final_x = info['torso_x']
        distance = final_x - initial_x
        
        # Episode statistics
        stats = {
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'distance': distance,
            'avg_loss': np.mean(episode_losses) if episode_losses else 0.0,
            'epsilon': self.agent.epsilon,
            'buffer_size': len(self.agent.memory),
            'terminated': done,
            'truncated': truncated
        }
        
        return stats
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate current agent performance."""
        results = evaluate_agent(
            agent=self.agent,
            env=self.env,
            num_episodes=num_episodes,
            render=False
        )
        
        return results
    
    def train(self):
        """Main training loop."""
        num_episodes = self.training_config['num_episodes']
        eval_freq = self.training_config['eval_freq']
        save_freq = self.training_config['save_freq']
        log_freq = self.training_config['log_freq']
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Evaluation every {eval_freq} episodes")
        print(f"Checkpoints every {save_freq} episodes\n")
        
        # Training loop with progress bar
        pbar = tqdm(range(num_episodes), desc="Training")
        
        for episode in pbar:
            # Train one episode
            stats = self.train_episode(episode)
            
            # Store statistics
            self.episode_rewards.append(stats['reward'])
            self.episode_lengths.append(stats['length'])
            self.episode_distances.append(stats['distance'])
            
            # Update progress bar
            pbar.set_postfix({
                'reward': f"{stats['reward']:.1f}",
                'dist': f"{stats['distance']:.2f}m",
                'eps': f"{stats['epsilon']:.3f}",
                'loss': f"{stats['avg_loss']:.4f}"
            })
            
            # Logging
            if episode % log_freq == 0:
                recent_rewards = self.episode_rewards[-log_freq:]
                recent_distances = self.episode_distances[-log_freq:]
                
                print(f"\n[Episode {episode}]")
                print(f"  Reward: {stats['reward']:.2f} (avg: {np.mean(recent_rewards):.2f})")
                print(f"  Distance: {stats['distance']:.2f}m (avg: {np.mean(recent_distances):.2f}m)")
                print(f"  Length: {stats['length']} steps")
                print(f"  Loss: {stats['avg_loss']:.4f}")
                print(f"  Epsilon: {stats['epsilon']:.4f}")
                print(f"  Buffer: {stats['buffer_size']}")
            
            # Evaluation
            if episode > 0 and episode % eval_freq == 0:
                print(f"\n{'='*50}")
                print(f"EVALUATION at Episode {episode}")
                print(f"{'='*50}")
                
                eval_results = self.evaluate(self.training_config['eval_episodes'])
                self.eval_rewards.append(eval_results['mean_reward'])
                
                print(f"  Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                print(f"  Mean distance: {eval_results['mean_distance']:.2f}m")
                print(f"  Mean length: {eval_results['mean_length']:.0f} steps")
                print(f"  Max reward: {eval_results['max_reward']:.2f}")
                print(f"{'='*50}\n")
                
                # Check for improvement
                if eval_results['mean_reward'] > self.best_eval_reward:
                    self.best_eval_reward = eval_results['mean_reward']
                    self.episodes_without_improvement = 0
                    
                    # Save best model
                    best_path = Path(self.training_config['checkpoint_dir']) / 'best_model.pth'
                    self.agent.save_checkpoint(str(best_path), episode, eval_results['mean_reward'])
                    print(f"   🏆 New best model! Reward: {self.best_eval_reward:.2f}\n")
                else:
                    self.episodes_without_improvement += eval_freq
                
                # Early stopping
                if self.training_config['early_stopping']:
                    if self.episodes_without_improvement >= self.training_config['patience']:
                        print(f"\n⚠ Early stopping: No improvement for {self.training_config['patience']} episodes")
                        break
                    
                    if self.best_eval_reward >= self.training_config['min_reward_threshold']:
                        print(f"\n✓ Success! Reached reward threshold: {self.best_eval_reward:.2f}")
                        break
            
            # Save checkpoint
            if episode > 0 and episode % save_freq == 0:
                checkpoint_path = Path(self.training_config['checkpoint_dir']) / f'checkpoint_ep{episode}.pth'
                self.agent.save_checkpoint(str(checkpoint_path), episode, stats['reward'])
        
        pbar.close()
        
        # Final evaluation
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")
        
        final_eval = self.evaluate(20)
        
        print(f"  Episodes trained: {self.agent.episodes_done}")
        print(f"  Best eval reward: {self.best_eval_reward:.2f}")
        print(f"  Final eval reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"  Final distance: {final_eval['mean_distance']:.2f}m")
        print(f"  Final length: {final_eval['mean_length']:.0f} steps")
        print(f"{'='*60}\n")
        
        # Save final model
        final_path = Path(self.training_config['checkpoint_dir']) / 'final_model.pth'
        self.agent.save_checkpoint(str(final_path), self.agent.episodes_done, final_eval['mean_reward'])
        
        # Save training statistics
        self.save_training_stats()
    
    def save_training_stats(self):
        """Save training statistics to file."""
        stats_path = Path(self.training_config['checkpoint_dir']) / 'training_stats.json'
        
        stats = {
            'num_episodes': self.agent.episodes_done,
            'best_eval_reward': float(self.best_eval_reward),
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'episode_lengths': [int(l) for l in self.episode_lengths],
            'episode_distances': [float(d) for d in self.episode_distances],
            'eval_rewards': [float(r) for r in self.eval_rewards],
            'env_config': self.env_config,
            'agent_config': self.agent_config,
            'training_config': self.training_config
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Training statistics saved to: {stats_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.env:
            self.env.close()


# Convenience functions

def train_agent(config_path: Optional[str] = None,
                env_config: Optional[Dict] = None,
                agent_config: Optional[Dict] = None,
                training_config: Optional[Dict] = None):
    """Train agent with given configurations."""
    # Create trainer
    trainer = Trainer(
        env_config=env_config,
        agent_config=agent_config,
        training_config=training_config
    )
    
    try:
        # Setup
        trainer.setup()
        
        # Train
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"\nTotal training time: {training_time/60:.1f} minutes")
        print(f"Time per episode: {training_time/trainer.agent.episodes_done:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving current progress...")
        
        # Save interrupted model
        interrupt_path = Path(trainer.training_config['checkpoint_dir']) / 'interrupted_model.pth'
        trainer.agent.save_checkpoint(
            str(interrupt_path),
            trainer.agent.episodes_done,
            trainer.episode_rewards[-1] if trainer.episode_rewards else 0.0
        )
        
        trainer.save_training_stats()
    
    finally:
        # Cleanup
        trainer.cleanup()
        print("\nTraining completed!")


def evaluate_trained_agent(checkpoint_path: str,
                           num_episodes: int = 10,
                           render: bool = False):
    """Evaluate a trained agent from checkpoint."""
    print("="*60)
    print("EVALUATING TRAINED AGENT")
    print("="*60)
    
    # Create environment
    env = HumanoidWalkEnvDiscrete(
        render_mode='human' if render else None,
        max_steps=1000,
        num_torque_bins=5
    )
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        num_joints=8,
        num_bins=5
    )
    
    # Load checkpoint
    agent.load_checkpoint(checkpoint_path)
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    
    # Evaluate
    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=num_episodes,
        render=render
    )
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean distance: {results['mean_distance']:.2f}m")
    print(f"  Mean length: {results['mean_length']:.0f} steps")
    print(f"  Max reward: {results['max_reward']:.2f}")
    print(f"  Min reward: {results['min_reward']:.2f}")
    print(f"{'='*60}\n")
    
    env.close()
    
    return results


if __name__ == "__main__":
    # Quick test training
    print("Running quick test training (10 episodes)...")
    
    train_agent(
        training_config={
            'num_episodes': 10,
            'eval_freq': 5,
            'save_freq': 5,
            'log_freq': 2,
            'checkpoint_dir': 'models/test',
            'use_pose_library': False,
            'early_stopping': False
        }
    )