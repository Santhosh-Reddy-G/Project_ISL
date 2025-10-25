import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple
import os
from pathlib import Path

# --- Constants for easy tuning ---
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = str(CURRENT_DIR/ "humanoid.urdf") # Assumes humanoid.urdf is in the project root
INITIAL_HEIGHT = 1.0  # Initial spawn height of the torso
TIME_STEP = 1.0 / 240.0  # Standard PyBullet physics step rate
FALL_THRESHOLD = 0.5    # Height below which the robot is considered fallen

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
            np.array([np.finfo(np.float32).max] * 2),      # Base Pos (x, z)
            np.array([1.0] * 4),                          # Base Orientation (Quaternion)
            action_high,                                  # Joint Positions
            np.array([np.finfo(np.float32).max] * self.action_dim) # Joint Velocities
        ])
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Cache for performance
        self.initial_joint_positions = self._get_initial_pose_vector()
        self.target_joint_positions = self.initial_joint_positions.copy()


    def _setup_joints(self) -> Tuple[list, int, int]:
        """Maps joint names to IDs and defines limits for the action space."""
        # Focus on 3-DoF Hips and 1-DoF Knees (6 joints total for action space)
        # Assuming the enhanced URDF uses the names from the previous step:
        JOINT_NAMES = [
            'right_hip_roll', 'right_hip_yaw', 'right_knee',  # Hip Pitch is implicit via thigh link
            'left_hip_roll', 'left_hip_yaw', 'left_knee',
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
        return motor_joints, len(motor_joints), len(motor_joints) * 2 + 6

    def _get_observation(self) -> np.ndarray:
        """Collects state data: base position/orientation and joint states."""
        
        # Base Link State (Torso)
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, self.client)
        # Only take X-position and Z-height (Y is side-to-side)
        base_state = np.array([pos[0], pos[2]] + list(ori)) 

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
        for _ in range(100):
            p.stepSimulation(self.client)

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    # ==================== 3. STEP FUNCTION ====================
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Applies action, steps physics, and calculates reward/termination.
        """
        
        # Apply action (target joint angles)
        self.target_joint_positions = action
        
        for i, (low, high, joint_id, force, max_vel) in enumerate(self.motor_joints):
            # Enforce action limits (though Gymnasium usually handles this)
            target_pos = np.clip(action[i], low, high)
            
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                # Crucial: Use the max_force from the URDF as the torque limit
                force=force, 
                maxVelocity=max_vel,
                physicsClientId=self.client
            )
        
        # Step physics simulation
        p.stepSimulation(self.client)
        
        # Calculate State and Reward
        observation = self._get_observation()
        reward, terminated = self._calculate_reward_and_termination(observation)
        
        # Truncated is typically used for reaching a maximum time limit
        truncated = False 
        info = self._get_info()
        
        if self.render_mode == "human":
            pass # Rendering is handled automatically by the GUI connection
            
        return observation, reward, terminated, truncated, info

    def _calculate_reward_and_termination(self, obs: np.ndarray) -> Tuple[float, bool]:
        """
        Defines the reward function and termination condition.
        Reward is the most critical component of the task!
        """
        
        # 1. Termination Check (Z-height of the torso)
        torso_z = obs[1] # Z-height is the second element of the observation vector
        terminated = torso_z < FALL_THRESHOLD
        
        # 2. Reward Calculation
        # Reward 1: Forward Progress (Maximize X-velocity)
        # We need the base velocity, which is not in the observation vector, so we get it live.
        torso_vel_lin, _ = p.getBaseVelocity(self.robot_id, self.client)
        forward_velocity = torso_vel_lin[0]
        
        # Reward 2: Survival Bonus (Encourages staying upright)
        survival_bonus = 1.0 if not terminated else 0.0
        
        # Reward 3: Control/Torque Penalty (Minimize motor effort)
        # This prevents the agent from jittering/using max torque needlessly.
        joint_velocities = obs[6 + self.action_dim: 6 + 2 * self.action_dim]
        # Quadratic penalty on velocity and small penalty on joint acceleration (approximated by velocity change)
        velocity_penalty = -0.05 * np.sum(np.square(joint_velocities))
        
        # Total Reward Weights (Tune these carefully)
        # +20 * forward_velocity (main goal)
        # +1.0 * survival_bonus
        # -0.05 * velocity_penalty
        
        reward = (20.0 * forward_velocity) + survival_bonus + velocity_penalty
        
        return float(reward), bool(terminated)

    def _get_info(self) -> Dict:
        """Returns diagnostic information for logging and debugging."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, self.client)
        torso_vel_lin, _ = p.getBaseVelocity(self.robot_id, self.client)
        
        return {
            "torso_x": pos[0],
            "torso_z": pos[2],
            "forward_velocity": torso_vel_lin[0],
            "action_dim": self.action_dim,
        }

    def close(self):
        """Disconnects the PyBullet client."""
        if self.client >= 0:
            p.disconnect(self.client)
            self.client = -1

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