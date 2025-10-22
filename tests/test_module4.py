"""
Test Suite for Module 4: Full Training Pipeline
Tests integration of all modules and training orchestration
FIXED FOR WINDOWS COMPATIBILITY
"""

import sys
import numpy as np
from pathlib import Path
import shutil
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import Trainer, train_agent, evaluate_trained_agent
from simulation.humanoid_env import load_pose_library


def safe_cleanup(directory):
    """Safely cleanup directory on Windows."""
    if Path(directory).exists():
        try:
            time.sleep(0.2)  # Wait for file handles to close
            shutil.rmtree(directory, ignore_errors=True)
            time.sleep(0.1)
        except Exception as e:
            print(f"  (Cleanup warning: {e})")
            pass


def test_trainer_initialization():
    """Test 1: Trainer initialization"""
    print("\n" + "="*60)
    print("TEST 1: Trainer Initialization")
    print("="*60)
    
    try:
        trainer = Trainer()
        
        print(f"✓ Trainer created")
        print(f"  Environment config: {trainer.env_config}")
        print(f"  Agent config keys: {list(trainer.agent_config.keys())}")
        print(f"  Training config keys: {list(trainer.training_config.keys())}")
        
        # Check directories created
        checkpoint_dir = Path(trainer.training_config['checkpoint_dir'])
        assert checkpoint_dir.exists(), "Checkpoint directory not created"
        print(f"✓ Checkpoint directory created: {checkpoint_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_setup():
    """Test 2: Trainer setup (environment, agent, pose library)"""
    print("\n" + "="*60)
    print("TEST 2: Trainer Setup")
    print("="*60)
    
    try:
        trainer = Trainer(
            training_config={'use_pose_library': False}  # Skip pose loading for speed
        )
        
        trainer.setup()
        
        # Check components initialized
        assert trainer.env is not None, "Environment not initialized"
        assert trainer.agent is not None, "Agent not initialized"
        
        print(f"✓ All components initialized")
        print(f"  Environment: {type(trainer.env).__name__}")
        print(f"  Agent: {type(trainer.agent).__name__}")
        print(f"  Pose library: {len(trainer.pose_library) if trainer.pose_library else 'None'}")
        
        trainer.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_episode():
    """Test 3: Single training episode"""
    print("\n" + "="*60)
    print("TEST 3: Single Training Episode")
    print("="*60)
    
    try:
        trainer = Trainer(
            training_config={'use_pose_library': False}
        )
        trainer.setup()
        
        print(f"Running single training episode...")
        
        stats = trainer.train_episode(episode=0)
        
        print(f"✓ Episode completed")
        print(f"  Reward: {stats['reward']:.2f}")
        print(f"  Length: {stats['length']} steps")
        print(f"  Distance: {stats['distance']:.2f}m")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Buffer size: {stats['buffer_size']}")
        
        # Verify stats structure
        required_keys = ['episode', 'reward', 'length', 'distance', 'epsilon']
        for key in required_keys:
            assert key in stats, f"Missing key in stats: {key}"
        
        print(f"✓ All stats keys present")
        
        trainer.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_short_training():
    """Test 4: Short training run (5 episodes)"""
    print("\n" + "="*60)
    print("TEST 4: Short Training Run (5 episodes)")
    print("="*60)
    
    test_dir = 'models/test_module4'
    
    try:
        # Clean up previous test
        safe_cleanup(test_dir)
        
        trainer = Trainer(
            training_config={
                'num_episodes': 5,
                'eval_freq': 3,
                'eval_episodes': 2,
                'save_freq': 3,
                'log_freq': 1,
                'checkpoint_dir': test_dir,
                'use_pose_library': False,
                'early_stopping': False
            }
        )
        
        trainer.setup()
        
        print(f"Starting training...")
        trainer.train()
        
        print(f"\n✓ Training completed")
        print(f"  Episodes: {len(trainer.episode_rewards)}")
        print(f"  Rewards: {trainer.episode_rewards}")
        print(f"  Best eval: {trainer.best_eval_reward:.2f}")
        
        # Check files created
        checkpoint_dir = Path(test_dir)
        assert checkpoint_dir.exists(), "Checkpoint directory not found"
        
        checkpoint_files = list(checkpoint_dir.glob('*.pth'))
        print(f"✓ Checkpoints saved: {len(checkpoint_files)}")
        
        stats_file = checkpoint_dir / 'training_stats.json'
        assert stats_file.exists(), "Training stats not saved"
        print(f"✓ Training stats saved")
        
        trainer.cleanup()
        time.sleep(0.2)  # Wait for cleanup
        
        # Clean up
        safe_cleanup(test_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        safe_cleanup(test_dir)
        return False


def test_pose_library_integration():
    """Test 5: Training with pose library"""
    print("\n" + "="*60)
    print("TEST 5: Pose Library Integration")
    print("="*60)
    
    pose_lib_path = Path('data/pose_library')
    
    if not pose_lib_path.exists():
        print(f"⚠ Pose library not found at {pose_lib_path}")
        print(f"  Run Module 1 first to generate pose library")
        print(f"  Skipping this test")
        return True
    
    try:
        # Load pose library
        pose_library = load_pose_library(str(pose_lib_path))
        
        print(f"✓ Loaded {len(pose_library)} poses")
        
        if len(pose_library) == 0:
            print(f"⚠ Pose library is empty")
            return True
        
        # Create trainer with pose library
        trainer = Trainer(
            training_config={
                'use_pose_library': True,
                'pose_library_path': str(pose_lib_path)
            }
        )
        
        trainer.setup()
        
        assert trainer.pose_library is not None, "Pose library not loaded"
        assert len(trainer.pose_library) > 0, "Pose library is empty"
        
        print(f"✓ Trainer loaded pose library")
        print(f"  Available poses: {list(trainer.pose_library.keys())[:5]}...")
        
        # Run one episode with random pose
        stats = trainer.train_episode(episode=0)
        
        print(f"✓ Episode with random pose completed")
        print(f"  Reward: {stats['reward']:.2f}")
        
        trainer.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """Test 6: Checkpoint saving and loading"""
    print("\n" + "="*60)
    print("TEST 6: Checkpoint Save/Load in Training")
    print("="*60)
    
    test_dir = 'models/test_checkpoint'
    
    try:
        # Clean up
        safe_cleanup(test_dir)
        
        trainer = Trainer(
            training_config={
                'num_episodes': 3,
                'save_freq': 2,
                'checkpoint_dir': test_dir,
                'use_pose_library': False,
                'early_stopping': False
            }
        )
        
        trainer.setup()
        trainer.train()
        
        # Check checkpoint exists
        checkpoint_dir = Path(test_dir)
        checkpoints = list(checkpoint_dir.glob('checkpoint_*.pth'))
        
        assert len(checkpoints) > 0, "No checkpoints saved"
        print(f"✓ Checkpoints saved: {len(checkpoints)}")
        
        # Try loading checkpoint
        checkpoint_path = checkpoints[0]
        trainer.agent.load_checkpoint(str(checkpoint_path))
        
        print(f"✓ Checkpoint loaded successfully")
        
        trainer.cleanup()
        time.sleep(0.2)
        
        # Clean up
        safe_cleanup(test_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        safe_cleanup(test_dir)
        return False


def test_evaluation():
    """Test 7: Agent evaluation"""
    print("\n" + "="*60)
    print("TEST 7: Agent Evaluation")
    print("="*60)
    
    try:
        trainer = Trainer(
            training_config={'use_pose_library': False}
        )
        trainer.setup()
        
        print(f"Evaluating untrained agent...")
        
        results = trainer.evaluate(num_episodes=3)
        
        print(f"✓ Evaluation completed")
        print(f"  Mean reward: {results['mean_reward']:.2f}")
        print(f"  Mean distance: {results['mean_distance']:.2f}m")
        print(f"  Mean length: {results['mean_length']:.0f}")
        
        # Check results structure
        required_keys = ['mean_reward', 'std_reward', 'mean_length', 'mean_distance']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        
        print(f"✓ All result keys present")
        
        trainer.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_agent_function():
    """Test 8: train_agent() convenience function"""
    print("\n" + "="*60)
    print("TEST 8: train_agent() Function")
    print("="*60)
    
    test_dir = 'models/test_train_func'
    
    try:
        # Clean up
        safe_cleanup(test_dir)
        
        print(f"Training with train_agent() function...")
        
        train_agent(
            training_config={
                'num_episodes': 3,
                'eval_freq': 2,
                'save_freq': 2,
                'log_freq': 1,
                'checkpoint_dir': test_dir,
                'use_pose_library': False,
                'early_stopping': False
            }
        )
        
        print(f"\n✓ train_agent() completed")
        
        # Check outputs
        checkpoint_dir = Path(test_dir)
        assert checkpoint_dir.exists(), "Checkpoint directory not created"
        
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        assert len(checkpoints) > 0, "No checkpoints saved"
        
        print(f"✓ Checkpoints created: {len(checkpoints)}")
        
        time.sleep(0.2)
        
        # Clean up
        safe_cleanup(test_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        safe_cleanup(test_dir)
        return False


def test_statistics_tracking():
    """Test 9: Training statistics tracking"""
    print("\n" + "="*60)
    print("TEST 9: Statistics Tracking")
    print("="*60)
    
    test_dir = 'models/test_stats'
    
    try:
        trainer = Trainer(
            training_config={
                'num_episodes': 5,
                'use_pose_library': False,
                'early_stopping': False,
                'checkpoint_dir': test_dir
            }
        )
        
        trainer.setup()
        
        # Run training
        for episode in range(5):
            stats = trainer.train_episode(episode)
            trainer.episode_rewards.append(stats['reward'])
            trainer.episode_lengths.append(stats['length'])
            trainer.episode_distances.append(stats['distance'])
        
        print(f"✓ Training completed")
        
        # Check statistics
        assert len(trainer.episode_rewards) == 5, "Wrong number of rewards"
        assert len(trainer.episode_lengths) == 5, "Wrong number of lengths"
        assert len(trainer.episode_distances) == 5, "Wrong number of distances"
        
        print(f"✓ Statistics tracking working")
        print(f"  Rewards: {[f'{r:.1f}' for r in trainer.episode_rewards]}")
        print(f"  Mean reward: {np.mean(trainer.episode_rewards):.2f}")
        print(f"  Mean distance: {np.mean(trainer.episode_distances):.2f}m")
        
        # Test stats saving
        trainer.save_training_stats()
        
        stats_file = Path(test_dir) / 'training_stats.json'
        assert stats_file.exists(), "Stats file not created"
        
        print(f"✓ Statistics saved to file")
        
        trainer.cleanup()
        time.sleep(0.2)
        
        # Clean up
        safe_cleanup(test_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        safe_cleanup(test_dir)
        return False


def run_all_tests():
    """Run all Module 4 tests"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#  MODULE 4 TEST SUITE - Full Training Pipeline" + " "*10 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    tests = [
        ("Trainer Initialization", test_trainer_initialization),
        ("Trainer Setup", test_trainer_setup),
        ("Single Episode", test_single_episode),
        ("Short Training", test_short_training),
        ("Pose Library Integration", test_pose_library_integration),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
        ("Evaluation", test_evaluation),
        ("train_agent() Function", test_train_agent_function),
        ("Statistics Tracking", test_statistics_tracking),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} | {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Module 4 is ready!")
        print("\n" + "="*60)
        print("PROJECT COMPLETE!")
        print("="*60)
        print("\nYou can now:")
        print("  1. Train the agent: python main.py --mode train")
        print("  2. Evaluate: python main.py --mode eval --checkpoint <path>")
        print("  3. Demo: python main.py --mode demo --checkpoint <path> --render")
        print("="*60)
    else:
        print("\n⚠ Some tests failed. Check errors above and fix.")
    
    print("\n" + "#"*60 + "\n")


if __name__ == "__main__":
    run_all_tests()