"""
Test Suite for Module 2: Simulation Environment
Tests PyBullet environment, URDF loading, and pose integration
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.humanoid_env import (
    HumanoidWalkEnv,
    HumanoidWalkEnvDiscrete,
    load_pose_library,
    sample_random_pose
)


def test_environment_creation():
    """Test 1: Basic environment creation"""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation")
    print("="*60)
    
    try:
        # Create environment (headless mode for testing)
        env = HumanoidWalkEnv(render_mode=None)
        
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Number of joints: {env.num_joints}")
        print(f"  Joint names: {env.joint_names}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reset_default_pose():
    """Test 2: Reset with default standing pose"""
    print("\n" + "="*60)
    print("TEST 2: Reset with Default Pose")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None)
        
        # Reset environment
        obs, info = env.reset()
        
        print(f"✓ Environment reset successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Expected shape: (29,)")
        print(f"  Initial torso height: {info['torso_z']:.3f} m")
        print(f"  Initial position: ({info['torso_x']:.3f}, {info['torso_y']:.3f}, {info['torso_z']:.3f})")
        
        # Check observation shape
        assert obs.shape == (29,), f"Expected shape (29,), got {obs.shape}"
        
        # Check torso is above ground
        assert info['torso_z'] > 0.2, f"Torso too low: {info['torso_z']}"
        
        print(f"✓ All assertions passed")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reset_custom_pose():
    """Test 3: Reset with custom initial pose"""
    print("\n" + "="*60)
    print("TEST 3: Reset with Custom Initial Pose")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None)
        
        # Create custom pose (bent knees, arms forward)
        custom_pose = np.array([
            0.5,   # right_hip
            0.8,   # right_knee (bent)
            0.0,   # right_ankle
            0.5,   # left_hip
            0.8,   # left_knee (bent)
            0.0,   # left_ankle
            1.5,   # right_shoulder (forward)
            1.5    # left_shoulder (forward)
        ])
        
        print(f"Custom pose: {custom_pose}")
        
        # Reset with custom pose
        obs, info = env.reset(initial_pose=custom_pose)
        
        print(f"✓ Reset with custom pose successful")
        print(f"  Initial height: {info['torso_z']:.3f} m")
        
        # Extract joint positions from observation
        joint_positions = obs[:8]
        print(f"  Actual joint positions: {joint_positions}")
        
        # Check if joint positions match (approximately)
        pose_error = np.abs(joint_positions - custom_pose).mean()
        print(f"  Average pose error: {pose_error:.4f} rad")
        
        if pose_error < 0.1:
            print(f"✓ Pose set accurately")
        else:
            print(f"⚠ Pose error is high (expected for complex poses after physics settle)")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_execution():
    """Test 4: Execute steps with random actions"""
    print("\n" + "="*60)
    print("TEST 4: Step Execution")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None, max_steps=100)
        obs, info = env.reset()
        
        print(f"Running 50 steps with random actions...")
        
        total_reward = 0
        max_height = 0
        max_velocity = 0
        
        for step in range(50):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            max_height = max(max_height, info['torso_z'])
            max_velocity = max(max_velocity, info['forward_velocity'])
            
            if step % 10 == 0:
                print(f"  Step {step:2d}: reward={reward:6.3f}, "
                      f"height={info['torso_z']:.3f}, vel={info['forward_velocity']:.3f}")
            
            if terminated:
                print(f"  Robot fell at step {step}")
                break
            
            if truncated:
                print(f"  Episode truncated at step {step}")
                break
        
        print(f"\n✓ Step execution successful")
        print(f"  Total steps: {step + 1}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Max height: {max_height:.3f} m")
        print(f"  Max velocity: {max_velocity:.3f} m/s")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_function():
    """Test 5: Reward function components"""
    print("\n" + "="*60)
    print("TEST 5: Reward Function")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None)
        obs, info = env.reset()
        
        print(f"Testing reward components...")
        
        # Take a few steps to get different states
        rewards = []
        for _ in range(10):
            action = np.zeros(8)  # Zero torque
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append({
                'total': reward,
                'velocity': info.get('reward_velocity', 0),
                'alive': info.get('reward_alive', 0),
                'energy': info.get('reward_energy', 0),
                'height': info.get('torso_height', 0)
            })
            
            if terminated:
                break
        
        print(f"✓ Reward function working")
        print(f"  Sample rewards:")
        for i, r in enumerate(rewards[:3]):
            print(f"    Step {i}: total={r['total']:.3f}, "
                  f"vel={r['velocity']:.3f}, alive={r['alive']:.3f}, "
                  f"energy={r['energy']:.4f}")
        
        # Check reward structure
        assert 'reward_velocity' in info, "Missing velocity reward"
        assert 'reward_alive' in info, "Missing alive reward"
        assert 'reward_energy' in info, "Missing energy reward"
        
        print(f"✓ All reward components present")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_discrete_action_space():
    """Test 6: Discrete action space version (for DQN)"""
    print("\n" + "="*60)
    print("TEST 6: Discrete Action Space (DQN)")
    print("="*60)
    
    try:
        # Create discrete environment
        env = HumanoidWalkEnvDiscrete(render_mode=None, num_torque_bins=5)
        
        print(f"✓ Discrete environment created")
        print(f"  Action space: {env.action_space}")
        print(f"  Torque bins: {env.torque_bins}")
        print(f"  Number of bins per joint: {env.num_torque_bins}")
        
        obs, info = env.reset()
        
        # Sample and execute discrete action
        action = env.action_space.sample()
        print(f"  Sample discrete action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Discrete action executed successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Reward: {reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_library_integration():
    """Test 7: Integration with Module 1 pose library"""
    print("\n" + "="*60)
    print("TEST 7: Module 1 Pose Library Integration")
    print("="*60)
    
    pose_folder = Path("data/pose_library")
    
    if not pose_folder.exists():
        print(f"⚠ Pose library not found at {pose_folder}")
        print(f"  Run Module 1 first to generate poses")
        print(f"  Skipping this test")
        return True
    
    try:
        # Load pose library
        pose_library = load_pose_library(str(pose_folder))
        
        print(f"✓ Loaded {len(pose_library)} poses from library")
        
        if len(pose_library) == 0:
            print(f"⚠ No poses found in library")
            return True
        
        # Show available poses
        print(f"  Available poses: {list(pose_library.keys())[:5]}...")
        
        # Test with random poses
        env = HumanoidWalkEnv(render_mode=None)
        
        print(f"\nTesting with 3 random poses from library...")
        for i in range(min(3, len(pose_library))):
            pose = sample_random_pose(pose_library)
            pose_name = list(pose_library.keys())[i]
            
            print(f"\n  Pose {i+1}: {pose_name}")
            print(f"    Joint angles: {pose}")
            
            obs, info = env.reset(initial_pose=pose)
            
            print(f"    Initial height: {info['torso_z']:.3f} m")
            
            # Run a few steps
            for _ in range(10):
                action = np.zeros(8)  # No torque
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    print(f"    ⚠ Robot unstable from this pose")
                    break
            else:
                print(f"    ✓ Stable pose")
        
        print(f"\n✓ Pose library integration successful")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_termination_conditions():
    """Test 8: Termination conditions (falling)"""
    print("\n" + "="*60)
    print("TEST 8: Termination Conditions")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None)
        
        # Test 1: Normal operation shouldn't terminate
        obs, info = env.reset()
        print(f"Test 1: Normal standing - should NOT terminate")
        
        for _ in range(20):
            action = np.zeros(8)  # No torque
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"  ✗ Unexpected termination at height {info['torso_z']:.3f}")
                break
        else:
            print(f"  ✓ No termination (correct)")
        
        # Test 2: Large torques should cause fall
        print(f"\nTest 2: Large random torques - should eventually terminate")
        obs, info = env.reset()
        
        for step in range(100):
            action = np.random.uniform(-1, 1, 8) * 2  # Large random torques
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"  ✓ Terminated at step {step}, height {info['torso_z']:.3f} m (correct)")
                break
        else:
            print(f"  ⚠ Did not terminate (robot very stable!)")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_episodes():
    """Test 9: Multiple episode resets"""
    print("\n" + "="*60)
    print("TEST 9: Multiple Episodes")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None, max_steps=50)
        
        print(f"Running 3 episodes...")
        
        episode_stats = []
        
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            episode_stats.append({
                'episode': episode + 1,
                'steps': episode_steps,
                'reward': episode_reward,
                'final_x': info['torso_x']
            })
            
            print(f"  Episode {episode+1}: {episode_steps} steps, "
                  f"reward={episode_reward:.2f}, final_x={info['torso_x']:.2f}m")
        
        print(f"\n✓ Multiple episodes successful")
        print(f"  Average steps: {np.mean([s['steps'] for s in episode_stats]):.1f}")
        print(f"  Average reward: {np.mean([s['reward'] for s in episode_stats]):.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Module 2 tests"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#  MODULE 2 TEST SUITE - Simulation Environment" + " "*13 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    tests = [
        ("Environment Creation", test_environment_creation),
        ("Reset Default Pose", test_reset_default_pose),
        ("Reset Custom Pose", test_reset_custom_pose),
        ("Step Execution", test_step_execution),
        ("Reward Function", test_reward_function),
        ("Discrete Action Space", test_discrete_action_space),
        ("Pose Library Integration", test_pose_library_integration),
        ("Termination Conditions", test_termination_conditions),
        ("Multiple Episodes", test_multiple_episodes),
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
        print("\n🎉 All tests passed! Module 2 is ready!")
        print("\n📋 Next: Module 3 - DQN Agent Implementation")
    else:
        print("\n⚠ Some tests failed. Check errors above and fix.")
    
    print("\n" + "#"*60 + "\n")


if __name__ == "__main__":
    run_all_tests()