import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple
import os
from pathlib import Path
import random

# --- Constants for easy tuning ---
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = str(CURRENT_DIR/ "humanoid.urdf") # Assumes humanoid.urdf is in the project root
INITIAL_HEIGHT = 1.0  # Initial spawn height of the torso
TIME_STEP = 1.0 / 240.0  # Standard PyBullet physics step rate
FALL_THRESHOLD = 0.5    # Height below which the robot is considered fallen
NUM_CONTROLLABLE_JOINTS = 8
NUM_BINS = 5
ACTION_TORQUES = np.linspace(-1.0, 1.0, NUM_BINS, dtype=np.float32)
JOINT_POSITION_GAIN = 0.05 # Step size for relative control
W_VEL = 20.0
W_LIVE = 0.5
W_ENERGY = 0.001
NEGATIVE_REWARD_FALL = -100.0

# --- Pose Library Integration Utility (Module 1 Compatibility) ---
def load_pose_library(path: str = "data/pose_library.npy") -> Dict[str, np.ndarray]:
    """
    Utility to load a pre-generated pose library for generalized training.
    Assumes the library is a dictionary saved as a NumPy file.
    """
    try:
        # Check if file exists, if not, create a default simple one
        if not Path(path).exists():
            print(f"WARNING: Pose library not found at {path}. Creating a dummy library.")
            # Default pose for 8 joints (must be aligned with the 8 joints setup)
            default_pose = np.zeros(NUM_CONTROLLABLE_JOINTS, dtype=np.float32)
            return {"default_stand": default_pose}
        
        # Load the actual file (assuming it's a dictionary of named poses)
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading pose library: {e}")
        return {"default_stand": np.zeros(NUM_CONTROLLABLE_JOINTS, dtype=np.float32)}
    
class HumanoidWalkEnv(gym.Env):
    """
    Custom Gymnasium environment for Humanoid Walking using PyBullet.
    Implements continuous action space for target joint angles.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # 1. Initialize PyBullet
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Configure physics engine
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(TIME_STEP, physicsClientId=self.client)
        
        # Load necessary objects (must happen before joint setup)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(URDF_PATH, [0, 0, INITIAL_HEIGHT], physicsClientId=self.client)
        
        # 2. Setup Joints and Spaces
        self.motor_joints, self.action_dim, self.obs_dim = self._setup_joints()
        
        # Define Spaces (Continuous Action Space is best for target control)
        # Action Space: Target angle for each motor joint (e.g., -1.57 to 1.57 rad)
        action_high = np.array([j[1] for j in self.motor_joints])
        action_low = np.array([j[0] for j in self.motor_joints])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation Space: [Base Pos (x, z), Base Quat (4), Joint Pos (N), Joint Vel (N)]
        # Where N is the number of controlled joints
        obs_high = np.concatenate([
            np.array([np.finfo(np.float32).max] * 3), # Base Pos (x, y, z) - FIX: Changed 2 to 3
            np.array([1.0] * 4), # Base Orientation (Quaternion)
            np.array([np.finfo(np.float32).max] * 3), # Base Linear Velocity (x, y, z) - ADDED
            np.array([np.finfo(np.float32).max] * 3), # Base Angular Velocity (x, y, z) - ADDED
            action_high,                                  # Joint Positions
            np.array([np.finfo(np.float32).max] * self.action_dim) # Joint Velocities
        ])
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Cache for performance
        self.initial_joint_positions = self._get_initial_pose_vector()
        self.target_joint_positions = self.initial_joint_positions.copy()

    def _setup_joints(self) -> Tuple[list, int, int]:
        """Maps joint names to IDs and defines limits for the action space. (Now 8 joints)"""
        # 6 Leg joints + 2 Arm joints = 8 total controllable joints
        JOINT_NAMES = [
            'right_hip_roll', 'right_hip_yaw', 'right_knee',  # 3 Leg joints
            'left_hip_roll', 'left_hip_yaw', 'left_knee',    # 3 Leg joints
            'right_shoulder_pitch', 'left_shoulder_pitch',  # 2 Arm joints (Assumed for 8-joint structure)
        ]
        
        # List of (low_limit, high_limit, joint_id, max_force, max_velocity)
        motor_joints = []
        
        for i in range(p.getNumJoints(self.robot_id, self.client)):
            info = p.getJointInfo(self.robot_id, i, self.client)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            if joint_name in JOINT_NAMES and joint_type == p.JOINT_REVOLUTE:
                low_limit, high_limit, max_force, max_vel = info[8], info[9], info[10], info[11]
                motor_joints.append((low_limit, high_limit, i, max_force, max_vel))
                
            elif joint_name == 'right_ankle_roll' or joint_name == 'left_ankle_roll':
                # Ankle joints are controlled but fixed for simplicity in this base class
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0,
                    force=10, # low force to keep them aligned
                    physicsClientId=self.client
                )

        print(f"Loaded {len(motor_joints)} controllable joints.")
        obs_dim = NUM_CONTROLLABLE_JOINTS * 2 + 13
        
        if len(motor_joints) != NUM_CONTROLLABLE_JOINTS:
             print(f"WARNING: Expected {NUM_CONTROLLABLE_JOINTS} joints, but loaded {len(motor_joints)}. Check URDF joint names.")
        
        return motor_joints, len(motor_joints), obs_dim

    def _get_observation(self) -> np.ndarray:
        """Collects state data: base position/orientation and joint states."""
        
        # Base Link State (Torso)
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, self.client)
        # Only take X-position and Z-height (Y is side-to-side)
        vel_lin, vel_ang = p.getBaseVelocity(self.robot_id, self.client)
        base_state = np.array([pos[0], pos[1], pos[2]] + list(ori) + list(vel_lin) + list(vel_ang))

        # Joint States
        joint_positions = []
        joint_velocities = []
        
        # Note: self.motor_joints contains (low, high, joint_id, ...)
        joint_ids = [j[2] for j in self.motor_joints]
        states = p.getJointStates(self.robot_id, joint_ids, self.client)
        
        for state in states:
            joint_positions.append(state[0])
            joint_velocities.append(state[1])
            
        return np.concatenate([base_state, joint_positions, joint_velocities], dtype=np.float32)

    def _get_initial_pose_vector(self) -> np.ndarray:
        """Returns the current joint positions as a vector (used for initial pose)."""
        joint_ids = [j[2] for j in self.motor_joints]
        states = p.getJointStates(self.robot_id, joint_ids, self.client)
        return np.array([s[0] for s in states])
    
    # ==================== 2. RESET FUNCTION ====================
    def reset(self, initial_pose: Optional[np.ndarray] = None, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the simulation, the robot's state, and applies an optional initial pose.
        """
        super().reset(seed=seed)
        
        # Reset PyBullet simulation and reload environment
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(URDF_PATH, [0, 0, INITIAL_HEIGHT], physicsClientId=self.client)
        
        # Re-get joint info after reloading the URDF
        self.motor_joints, _, _ = self._setup_joints() 
        
        if initial_pose is None:
            # Use the default standing pose
            target_pose = self.initial_joint_positions
        else:
            # Provided pose must match the action dimension
            assert initial_pose.shape[0] == self.action_dim, "Initial pose must match the action dimension."
            target_pose = initial_pose
            
        self.target_joint_positions = target_pose.copy()

        # Iterate through controllable joints and reset their state
        for i, joint_id in enumerate([j[2] for j in self.motor_joints]):
            # Set joint position and velocity to the target pose
            p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_id,
                targetValue=target_pose[i],
                targetVelocity=0,
                physicsClientId=self.client
            )
            
            # Immediately apply motors to the reset position
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pose[i],
                force=self.motor_joints[i][3], # Use the max_force defined in URDF
                maxVelocity=self.motor_joints[i][4],
                physicsClientId=self.client
            )
        
        # Take one step to settle the robot on the ground plane
        #for _ in range(100):
            p.stepSimulation(self.client)

        observation = self._get_observation()
        info = self._get_info(observation)
        return observation, info
    
    # ==================== 3. STEP FUNCTION ====================
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Applies action, steps physics, and calculates reward/termination.
        """
        
        # --- 1. Pre-computation and Clamping (Optimized) ---
        # Action limits should ideally be calculated once in __init__
        action_high = np.array([j[1] for j in self.motor_joints], dtype=np.float32)
        action_low = np.array([j[0] for j in self.motor_joints], dtype=np.float32)

        # Clip the action to enforce joint limits
        self.target_joint_positions = np.clip(action, action_low, action_high)
        
        applied_torques_proxy = np.zeros(self.action_dim, dtype=np.float32)
        current_joint_pos = self._get_initial_pose_vector()

        # --- 2. Apply Motor Control (Single Loop) ---
        for i, (_, _, joint_id, max_force, max_vel) in enumerate(self.motor_joints):
            target_pos = self.target_joint_positions[i]
            
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=max_force,
                maxVelocity=max_vel,
                physicsClientId=self.client
            )
            
            # Proxy: Deviation of the target position from the current position
            applied_torques_proxy[i] = target_pos - current_joint_pos[i]
        
        # --- 3. Step Physics ---
        p.stepSimulation(self.client)
        
        # --- 4. Calculate State and Reward ---
        observation = self._get_observation()
        
        # FIX: Correctly unpack 3 values (reward, terminated, info_reward)
        reward, terminated, info_reward = self._calculate_reward_and_termination(observation, applied_torques_proxy)
        
        # Truncated is typically used for reaching a maximum time limit
        truncated = False 
        
        # Merge Info: Get base info, then update with reward/termination diagnostics
        info = self._get_info(observation) # <-- This now correctly calls the method with 2 arguments
        info.update(info_reward) 

        if self.render_mode == "human":
            pass 
            
        return observation, reward, terminated, truncated, info
    # CORRECT definition:
    
    def _calculate_reward_and_termination(self, obs: np.ndarray,torque_proxy: np.ndarray) -> Tuple[float, bool, Dict]:
        """
        Defines the reward function and termination condition.
        Reward is the most critical component of the task!
        """
        
        # 1. Termination Check (Z-height of the torso)
        torso_z = obs[2] # Z-height is the second element of the observation vector
        terminated = torso_z < FALL_THRESHOLD
        
        # 2. Reward Components (using new 29-dim observation indices)
        
        # rvel: Forward velocity (X-velocity, index 7)
        forward_velocity = obs[7]
        r_vel = forward_velocity
        
        # rlive: Small, constant positive reward for survival
        r_live = 1.0 
        
        # renergy: Penalty for the sum of squared torques (proxied by position deviation)
        r_energy = np.sum(torque_proxy**2)
        
        # Rt = wvel · rvel + wlive · rlive − wenergy · renergy
        reward = (W_VEL * r_vel) + (W_LIVE * r_live) - (W_ENERGY * r_energy)
        
        # 3. Apply large negative reward on termination
        if terminated:
            reward += NEGATIVE_REWARD_FALL 
        
        # 4. Info Dict Update
        info = {
            "torso_x": obs[0],
            "torso_y": obs[1], # ADDED Y-position
            "torso_z": torso_z,
            "forward_velocity": forward_velocity,
            "action_dim": self.action_dim,
        }
        
        return float(reward), bool(terminated), info

    def _get_info(self, obs: np.ndarray) -> Dict: # <--- FIX: MUST include 'self' AND 'obs'
        """Returns diagnostic information for logging and debugging based on the observation vector."""
        # Use the first elements of the 29-dim observation vector
        return {
            "torso_x": obs[0],
            "torso_y": obs[1],
            "torso_z": obs[2],
            "forward_velocity": obs[7], # X-velocity
            "action_dim": self.action_dim,
        }

    def close(self):
        """Disconnects the PyBullet client."""
        if self.client >= 0:
            p.disconnect(self.client)
            self.client = -1
class HumanoidWalkEnvDiscrete(HumanoidWalkEnv):
        """
        Gymnasium environment with a discrete action space wrapper for DQN.
        Action space is Box(low=0, high=4, shape=(8,)) to match the [8, 5] output 
        of the Q-network where argmax is taken for each joint.
        """
    
        def __init__(self, render_mode: Optional[str] = None, max_steps: int = 500):
            super().__init__(render_mode=render_mode)
        
            self.max_steps = max_steps
            self.current_step = 0
            self.num_joints = self.action_dim 
            self.num_bins = NUM_BINS
        
            # Action Space: 8 joints, each choosing one of 5 bins (0 to 4)
            # This is compatible with the "multi-head" Q-network output and argmax per joint.
            self.action_space = spaces.Box(
                low=0, 
                high=self.num_bins - 1, 
                shape=(self.num_joints,), 
                dtype=np.int32
            )

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            """
            Takes an array of 8 discrete indices (0-4) and translates it into 
            8 continuous position targets (relative control).
            """
        
            # 1. Decode the discrete action (8 integers) into 8 continuous delta positions
            # The indices (0-4) map to the ACTION_TORQUES array
            delta_angles = ACTION_TORQUES[action.astype(np.int32)]
        
            # Update target positions (relative control)
            self.target_joint_positions += delta_angles * JOINT_POSITION_GAIN
        
            # 2. Clamp targets to the continuous joint limits
            action_low = np.array([j[0] for j in self.motor_joints])
            action_high = np.array([j[1] for j in self.motor_joints])
        
            # Pass the clipped continuous targets to the parent's step function
            obs, reward, terminated, truncated_parent, info = super().step(
                self.target_joint_positions 
            )
        
            # 3. Handle max steps (Truncation)
            self.current_step += 1
            truncated = self.current_step >= self.max_steps
        
            return obs, reward, terminated, truncated, info

        def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
            obs, info = super().reset(**kwargs)
            self.current_step = 0
            return obs, info
        
# Example usage (for testing the environment)
if __name__ == '__main__':
    # This assumes you have the humanoid_biped.urdf file available
    env = HumanoidWalkEnv(render_mode='human')
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Initial Observation Dim: {obs.shape}")
    
    for episode in range(3):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        for step in range(500):
            # Take a random action for testing
            action = env.action_space.sample() 
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episode {episode+1} finished after {step+1} steps with reward: {total_reward:.2f}")
                break
                
    env.close()