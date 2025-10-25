"""
Test Suite for Module 2: Simulation Environment (HumanoidWalkEnv)
Tests PyBullet environment, URDF loading, and pose integration.
"""

import sys
import numpy as np
from pathlib import Path
import time
import gymnasium as gym

# Add parent directory to path, assuming tests/test_module2.py structure
# This allows the script to find the 'simulation' package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the environment class
from simulation.humanoid_env import HumanoidWalkEnv
from simulation.humanoid_env import HumanoidWalkEnvDiscrete
# Expected dimensions derived from HumanoidWalkEnv:
# ACTION_DIM = 8 (6 controllable hip/knee + 2 arm joints)
# OBS_DIM = [Base Pos (3) + Base Quat (4) + Base Lin Vel (3) + Base Ang Vel (3) + Joint Pos (8) + Joint Vel (8)] = 29
ACTION_DIM = 8
OBS_DIM = 29 
FALL_THRESHOLD = 0.5 # From the environment definition

def test_environment_creation():
    """Test 1: Basic environment creation and space dimensions."""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation & Dimension Check")
    print("="*60)
    
    try:
        # Create environment (headless mode for testing)
        env = HumanoidWalkEnv(render_mode=None)
        
        print(f"✓ Environment created successfully")
        print(f"  Expected Obs. Dim: {OBS_DIM}, Actual: {env.observation_space.shape[0]}")
        print(f"  Expected Act. Dim: {ACTION_DIM}, Actual: {env.action_space.shape[0]}")
        
        # Check dimensions
        assert env.observation_space.shape == (OBS_DIM,), f"Obs shape expected ({OBS_DIM},), got {env.observation_space.shape}"
        assert env.action_space.shape == (ACTION_DIM,), f"Action shape expected ({ACTION_DIM},), got {env.action_space.shape}"
        
        env.close()
        print("✓ All assertions passed")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------------------------------------------

def test_reset_default_pose():
    """Test 2: Reset with default standing pose and initial height check."""
    print("\n" + "="*60)
    print("TEST 2: Reset with Default Pose (Height Check)")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None)
        
        # Reset environment
        obs, info = env.reset()
        
        print(f"✓ Environment reset successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial torso height: {info['torso_z']:.3f} m")
        
        # Check observation shape
        assert obs.shape == (OBS_DIM,), f"Expected shape ({OBS_DIM},), got {obs.shape}"
        
        # Torso Z-height check (must be above the falling threshold)
        assert info['torso_z'] > FALL_THRESHOLD, f"Torso height {info['torso_z']:.3f} is too low. Check INITIAL_HEIGHT in humanoid_env.py."
        
        env.close()
        print(f"✓ All assertions passed")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------------------------------------------

def test_reset_custom_pose():
    """Test 3: Reset with custom pose and verification of joint angles."""
    print("\n" + "="*60)
    print("TEST 3: Reset with Custom Initial Pose (Joint Check)")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None)
        
        # Custom pose (ACTION_DIM = 8): e.g., slightly crouched, wide stance
        # Order: R_Hip_Roll, R_Hip_Yaw, R_Knee, L_Hip_Roll, L_Hip_Yaw, L_Knee
        custom_pose = np.array([
            0.2,    # R_Hip_Roll (outward)
            0.0,    # R_Hip_Yaw (was R_Hip_Yaw, but order might be R_Hip_Roll, R_Hip_Pitch, R_Knee, L_Hip_Roll, L_Hip_Pitch, L_Knee)
            0.8,    # R_Knee (bent significantly)
            -0.2,   # L_Hip_Roll (inward)
            0.0,    # L_Hip_Yaw
            0.8,    # L_Knee (bent significantly)
            0.3,    # R_Shoulder_Pitch  <-- ADDED ARM JOINTS
            0.3     # L_Shoulder_Pitch  <-- ADDED ARM JOINTS
        ], dtype=np.float32)
        
        assert custom_pose.shape[0] == ACTION_DIM, f"Custom pose must be size {ACTION_DIM}, got {custom_pose.shape[0]}"
        
        # Reset with custom pose
        obs, info = env.reset(initial_pose=custom_pose)
        
        print(f"✓ Reset with custom pose successful")
        
        # Joint positions start after:
        # 3 (Base Pos) + 4 (Base Quat) + 3 (Base Lin Vel) + 3 (Base Ang Vel) = 13
        # Index starts at 0, so the joint positions start at index 13.
        joint_positions = obs[13:13 + ACTION_DIM] # <-- CHANGE: from obs[6:6 + ACTION_DIM] to obs[13:13 + ACTION_DIM]
        
        # Check if joint positions match (allowing for small float tolerance after simulation step)
        pose_error = np.abs(joint_positions - custom_pose).max()
        
        print(f"  Custom Pose (Target): {custom_pose}")
        print(f"  Actual Joint Pos (Obs): {joint_positions}")
        print(f"  Maximum pose error: {pose_error:.6f} rad")
        
        # PyBullet resetJointState is usually very accurate
        assert pose_error < 1e-3, f"Pose error is too high ({pose_error:.6f})"
        
        env.close()
        print(f"✓ All assertions passed")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------------------------------------------

def test_step_execution():
    """Test 4: Execute steps with random actions."""
    print("\n" + "="*60)
    print("TEST 4: Step Execution")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None) 
        obs, info = env.reset()
        
        print(f"Running 20 steps with random actions (Action Dim: {ACTION_DIM})...")
        
        for step in range(20):
            # Sample action within the continuous space limits
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Basic type and shape checks
            assert isinstance(reward, float) or isinstance(reward, np.floating), f"Reward is not a float: {type(reward)}"
            assert obs.shape == (OBS_DIM,), f"Observation shape mismatch: {obs.shape}"
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            
            if step % 5 == 0:
                print(f"  Step {step:2d}: reward={reward:6.3f}, "
                      f"height={info['torso_z']:.3f}, vel={info['forward_velocity']:.3f}, Terminated={terminated}")
            
            if terminated:
                print(f"  Episode ended early at step {step} (Height: {info['torso_z']:.3f}m)")
                break
        
        env.close()
        print(f"✓ Step execution successful and basic outputs verified")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------------------------------------------

def test_termination_conditions():
    """Test 5: Termination conditions (falling below threshold)."""
    print("\n" + "="*60)
    print("TEST 5: Termination Conditions")
    print("="*60)
    
    try:
        env = HumanoidWalkEnv(render_mode=None)
        
        # Test 1: Normal standing - should NOT terminate immediately
        obs, info = env.reset()
        print(f"Test 1: Normal standing (Initial height: {info['torso_z']:.3f}m)")
        
        for _ in range(5):
            action = np.zeros(ACTION_DIM) 
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"  ✗ Unexpected termination at height {info['torso_z']:.3f}")
                break
        else:
            print(f"  ✓ No immediate termination (correct)")
        
        # Test 2: Induce a fall (by trying to force the robot to a low, unstable pose)
        print(f"\nTest 2: Induce a fall to check termination threshold ({FALL_THRESHOLD}m)")
        obs, info = env.reset()
        
        termination_occurred = False
        
        # FIX: Define a highly unstable, asymmetric action for guaranteed fall (8-dim vector)
        action_high = env.action_space.high
        action_low = env.action_space.low
        
        unstable_action = np.zeros(ACTION_DIM, dtype=np.float32)
        
        # Maximize the sideways lean and bend the knees
        unstable_action[0] = action_high[0] * 0.9 # Right Hip Roll (Max Out)
        unstable_action[3] = action_low[3] * 0.9  # Left Hip Roll (Max Inward/Negative Roll)
        unstable_action[2] = action_high[2] * 0.9 # Right Knee (Bend)
        unstable_action[5] = action_high[5] * 0.9 # Left Knee (Bend)
        # Set arms to a wide, unstable pose
        unstable_action[6] = action_high[6] * 0.8
        unstable_action[7] = action_low[7] * 0.8 
        
        # FIX: Increase simulation steps from 100 to 500 (approx 2 seconds) for a guaranteed fall
        for step in range(500): 
            obs, reward, terminated, truncated, info = env.step(unstable_action)
            
            if terminated:
                print(f"  ✓ Terminated at step {step}, final height {info['torso_z']:.3f} m (Correct, Z < {FALL_THRESHOLD}m)")
                termination_occurred = True
                break
        
        assert termination_occurred, "Robot did not terminate when unstable actions were applied."
        
        env.close()
        print(f"✓ All assertions passed")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------------------------------------------

def run_all_tests():
    """Run all Module 2 tests and summarize results."""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#  MODULE 2 TEST SUITE - Simulation Environment (HumanoidWalkEnv)"+" "*2 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    tests = [
        ("Test 1: Environment Creation", test_environment_creation),
        ("Test 2: Reset Default Pose", test_reset_default_pose),
        ("Test 3: Reset Custom Pose", test_reset_custom_pose),
        ("Test 4: Step Execution", test_step_execution),
        ("Test 5: Termination Conditions", test_termination_conditions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed unexpectedly: {e}")
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
        print("\n🎉 All core simulation tests passed!")
    else:
        print("\n⚠ Some tests failed. Check the errors above and fix the environment code.")
    
    print("\n" + "#"*60 + "\n")

if __name__ == "__main__":
    run_all_tests()