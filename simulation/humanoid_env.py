"""
Module 2: Humanoid Walking Environment
Gymnasium-compliant PyBullet environment for humanoid locomotion
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class HumanoidWalkEnv(gym.Env):
    """
    Gymnasium environment for humanoid walking with initial pose support.
    
    Observation Space:
        - Joint positions (8)
        - Joint velocities (8)
        - Torso position (3)
        - Torso orientation (4 quaternion)
        - Torso linear velocity (3)
        - Torso angular velocity (3)
        Total: 29 dimensions
    
    Action Space:
        - Joint torques for 8 joints: [-1, 1] normalized
    
    Reward:
        - Forward velocity (primary)
        - Alive bonus
        - Energy penalty
        - Fall penalty
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, 
                 urdf_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 max_steps: int = 1000,
                 render_fps: int = 60):
        """
        Initialize the humanoid walking environment.
        
        Args:
            urdf_path: Path to humanoid URDF file
            render_mode: "human" for GUI, "rgb_array" for headless, None for fastest
            max_steps: Maximum steps per episode
            render_fps: Rendering FPS
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.render_fps = render_fps
        
        # Find URDF path
        if urdf_path is None:
            # Try to find in simulation folder
            urdf_path = Path(__file__).parent / "humanoid.urdf"
            if not urdf_path.exists():
                raise FileNotFoundError(
                    f"URDF file not found at {urdf_path}. "
                    "Please provide urdf_path or place humanoid.urdf in simulation folder."
                )
        
        self.urdf_path = str(urdf_path)
        
        # Joint names matching your Module 1 output
        self.joint_names = [
            'right_hip', 'right_knee', 'right_ankle',
            'left_hip', 'left_knee', 'left_ankle',
            'right_shoulder', 'left_shoulder'
        ]
        
        self.num_joints = len(self.joint_names)
        
        # Observation space: [joint_pos(8), joint_vel(8), torso_pos(3), 
        #                     torso_orn(4), torso_lin_vel(3), torso_ang_vel(3)]
        obs_dim = 29
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: normalized torques [-1, 1] for each joint
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # Max torques for each joint (will be scaled by actions)
        self.max_torques = np.array([100, 100, 50, 100, 100, 50, 50, 50])
        
        # Reward weights
        self.w_vel = 1.0      # Forward velocity weight
        self.w_live = 0.1     # Alive bonus weight
        self.w_energy = 0.001 # Energy penalty weight
        self.fall_penalty = -10.0
        
        # Physics parameters
        self.dt = 1.0 / 240.0  # PyBullet default timestep
        self.frame_skip = 4     # Execute action for 4 physics steps
        
        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        
        # Initialize PyBullet
        self.physics_client = None
        self.robot_id = None
        self.joint_indices = {}
        self.initial_pose = None
        
        self._setup_pybullet()
    
    def _setup_pybullet(self):
        """Initialize PyBullet physics engine."""
        # Connect to PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set ground friction
        p.changeDynamics(
            self.plane_id, -1,
            lateralFriction=1.0,
            spinningFriction=0.1,
            rollingFriction=0.1
        )
    
    def _load_robot(self):
        """Load humanoid robot from URDF."""
        # Load robot at default standing height
        start_pos = [0, 0, 1.0]  # Start 1 meter above ground
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        self.robot_id = p.loadURDF(
            self.urdf_path,
            start_pos,
            start_orientation,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # Get joint indices
        self.joint_indices = {}
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name in self.joint_names:
                self.joint_indices[joint_name] = i
        
        # Set foot friction
        for joint_name in ['right_ankle', 'left_ankle']:
            if joint_name in self.joint_indices:
                foot_link = self.joint_indices[joint_name] + 1
                p.changeDynamics(
                    self.robot_id, foot_link,
                    lateralFriction=2.0,
                    spinningFriction=0.1,
                    rollingFriction=0.1
                )
    
    def reset(self, 
              seed: Optional[int] = None,
              options: Optional[Dict] = None,
              initial_pose: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            initial_pose: Initial joint angles (8,) from Module 1
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.step_count = 0
        self.episode_count += 1
        
        # Store initial pose for this episode
        self.initial_pose = initial_pose
        
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        # Reload environment
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)
        self._load_robot()
        
        # Set initial pose if provided
        if initial_pose is not None:
            self._set_initial_pose(initial_pose)
        else:
            # Default standing pose
            default_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self._set_initial_pose(default_pose)
        
        # Let robot settle
        for _ in range(10):
            p.stepSimulation()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _set_initial_pose(self, joint_angles: np.ndarray):
        """
        Set robot to initial pose from joint angles.
        
        Args:
            joint_angles: Array of 8 joint angles (radians)
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joint angles, got {len(joint_angles)}"
            )
        
        # Set each joint to specified angle
        for joint_name, angle in zip(self.joint_names, joint_angles):
            if joint_name in self.joint_indices:
                joint_idx = self.joint_indices[joint_name]
                
                # Reset joint state (position and velocity)
                p.resetJointState(
                    self.robot_id,
                    joint_idx,
                    targetValue=angle,
                    targetVelocity=0.0
                )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Normalized torques [-1, 1] for each joint
        
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode ended (fell)
            truncated: Whether episode hit max steps
            info: Additional information
        """
        # Clip and scale actions to actual torques
        action = np.clip(action, -1.0, 1.0)
        torques = action * self.max_torques
        
        # Apply torques to joints
        for joint_name, torque in zip(self.joint_names, torques):
            if joint_name in self.joint_indices:
                joint_idx = self.joint_indices[joint_name]
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.TORQUE_CONTROL,
                    force=torque
                )
        
        # Step physics multiple times (frame skip)
        for _ in range(self.frame_skip):
            p.stepSimulation()
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward, reward_info = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        
        # Update step count
        self.step_count += 1
        
        # Prepare info dict
        info = self._get_info()
        info.update(reward_info)
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation of the environment.
        
        Returns:
            Observation vector (29,)
        """
        # Joint positions and velocities
        joint_positions = np.zeros(self.num_joints)
        joint_velocities = np.zeros(self.num_joints)
        
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in self.joint_indices:
                joint_idx = self.joint_indices[joint_name]
                joint_state = p.getJointState(self.robot_id, joint_idx)
                joint_positions[i] = joint_state[0]  # Position
                joint_velocities[i] = joint_state[1]  # Velocity
        
        # Torso (base link) state
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)
        torso_lin_vel, torso_ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Concatenate all observations
        obs = np.concatenate([
            joint_positions,      # 8
            joint_velocities,     # 8
            torso_pos,           # 3
            torso_orn,           # 4 (quaternion)
            torso_lin_vel,       # 3
            torso_ang_vel        # 3
        ])
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate reward for current step.
        
        Args:
            action: Action taken
        
        Returns:
            reward: Total reward
            reward_info: Breakdown of reward components
        """
        # Get torso velocity
        torso_lin_vel, _ = p.getBaseVelocity(self.robot_id)
        forward_vel = torso_lin_vel[0]  # X-axis is forward
        
        # Get torso height
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        torso_height = torso_pos[2]
        
        # Reward components
        r_vel = forward_vel  # Encourage forward movement
        r_live = 1.0 if torso_height > 0.3 else 0.0  # Alive bonus
        r_energy = -np.sum(action ** 2)  # Penalize energy use
        
        # Total reward
        reward = (
            self.w_vel * r_vel +
            self.w_live * r_live +
            self.w_energy * r_energy
        )
        
        # Add fall penalty if terminated
        if self._is_terminated():
            reward += self.fall_penalty
        
        reward_info = {
            'reward_velocity': r_vel,
            'reward_alive': r_live,
            'reward_energy': r_energy,
            'forward_velocity': forward_vel,
            'torso_height': torso_height
        }
        
        return reward, reward_info
    
    def _is_terminated(self) -> bool:
        """
        Check if episode should terminate (robot fell).
        
        Returns:
            True if robot fell
        """
        # Get torso position and orientation
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Check height (fell if torso too low)
        if torso_pos[2] < 0.3:
            return True
        
        # Check orientation (fell if tilted too much)
        euler = p.getEulerFromQuaternion(torso_orn)
        pitch = abs(euler[1])
        roll = abs(euler[0])
        
        if pitch > np.pi/3 or roll > np.pi/3:  # More than 60 degrees tilt
            return True
        
        return False
    
    def _get_info(self) -> Dict:
        """
        Get additional information about current state.
        
        Returns:
            Dictionary with info
        """
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)
        torso_lin_vel, torso_ang_vel = p.getBaseVelocity(self.robot_id)
        
        return {
            'step': self.step_count,
            'episode': self.episode_count,
            'torso_x': torso_pos[0],
            'torso_y': torso_pos[1],
            'torso_z': torso_pos[2],
            'forward_velocity': torso_lin_vel[0],
        }
    
    def render(self):
        """Render environment (handled by PyBullet GUI if enabled)."""
        if self.render_mode == "human":
            # GUI is already rendering
            pass
        elif self.render_mode == "rgb_array":
            # Get camera image
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=3.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width/height,
                nearVal=0.1, farVal=100.0
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
            return rgb_array
    
    def close(self):
        """Clean up environment."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


class HumanoidWalkEnvDiscrete(HumanoidWalkEnv):
    """
    Discrete action space version for DQN.
    
    Each joint can take one of N torque values (e.g., 5 bins).
    Total actions = num_joints * num_bins
    """
    
    def __init__(self,
                 urdf_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 max_steps: int = 1000,
                 num_torque_bins: int = 5):
        """
        Initialize discrete action space environment.
        
        Args:
            urdf_path: Path to humanoid URDF
            render_mode: Render mode
            max_steps: Max steps per episode
            num_torque_bins: Number of discrete torque values per joint
        """
        super().__init__(urdf_path, render_mode, max_steps)
        
        self.num_torque_bins = num_torque_bins
        
        # Discrete action space: one action per joint
        # Action value selects which torque bin
        # Total actions = num_joints * num_torque_bins
        self.action_space = spaces.MultiDiscrete(
            [num_torque_bins] * self.num_joints
        )
        
        # Define torque bins (e.g., for 5 bins: [-1.0, -0.5, 0, 0.5, 1.0])
        self.torque_bins = np.linspace(-1.0, 1.0, num_torque_bins)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute discrete action.
        
        Args:
            action: Array of discrete indices [0, num_bins-1] for each joint
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert discrete actions to continuous torques
        continuous_action = np.array([
            self.torque_bins[int(a)] for a in action
        ])
        
        # Call parent step with continuous action
        return super().step(continuous_action)


# Utility functions for loading poses from Module 1

def load_pose_library(pose_folder: str) -> Dict[str, np.ndarray]:
    """
    Load all poses from Module 1 pose library.
    
    Args:
        pose_folder: Path to pose library folder
    
    Returns:
        Dictionary mapping pose names to joint angles
    """
    from pathlib import Path
    
    pose_folder = Path(pose_folder)
    pose_library = {}
    
    # Load all .npy files
    for npy_file in pose_folder.glob("*.npy"):
        pose_name = npy_file.stem
        joint_angles = np.load(npy_file)
        
        # Validate shape
        if joint_angles.shape == (8,):
            pose_library[pose_name] = joint_angles
        else:
            print(f"Warning: Skipping {pose_name}, invalid shape {joint_angles.shape}")
    
    return pose_library


def sample_random_pose(pose_library: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Sample a random pose from the library.
    
    Args:
        pose_library: Dictionary of poses
    
    Returns:
        Random joint angle vector
    """
    import random
    pose_name = random.choice(list(pose_library.keys()))
    return pose_library[pose_name]


# Example usage
if __name__ == "__main__":
    print("Testing HumanoidWalkEnv...")
    
    # Create environment
    env = HumanoidWalkEnv(render_mode="human")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test with default pose
    print("\nTest 1: Default standing pose")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few steps with random actions
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.3f}, height={info['torso_z']:.3f}, "
                  f"vel={info['forward_velocity']:.3f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break
    
    # Test with custom pose
    print("\nTest 2: Custom initial pose")
    custom_pose = np.array([0.5, 0.3, 0.0, 0.5, 0.3, 0.0, 1.0, 1.0])
    obs, info = env.reset(initial_pose=custom_pose)
    print(f"Reset with custom pose: {custom_pose}")
    
    env.close()
    print("\n✓ Environment tests passed!")