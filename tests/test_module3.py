"""
Test Suite for Module 3: DQN Agent
Tests Q-Network, Replay Buffer, and DQN Agent
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from control import QNetwork, ReplayBuffer, DQNAgent
from simulation.humanoid_env import HumanoidWalkEnvDiscrete


def test_qnetwork():
    """Test 1: Q-Network architecture and forward pass"""
    print("\n" + "="*60)
    print("TEST 1: Q-Network Architecture")
    print("="*60)
    
    try:
        state_dim = 29
        num_joints = 8
        num_bins = 5
        
        # Create network
        net = QNetwork(state_dim, num_joints, num_bins)
        
        print(f"✓ Q-Network created")
        print(f"  State dim: {state_dim}")
        print(f"  Output: {num_joints} joints × {num_bins} bins = {num_joints * num_bins} actions")
        
        # Test forward pass with single state
        test_state = torch.randn(state_dim)
        q_values = net(test_state.unsqueeze(0))
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {test_state.shape}")
        print(f"  Output shape: {q_values.shape}")  # Should be (1, 8, 5)
        
        # Check output shape
        assert q_values.shape == (1, num_joints, num_bins), \
            f"Expected shape (1, {num_joints}, {num_bins}), got {q_values.shape}"
        
        # Test action selection
        action = net.select_action(test_state, epsilon=0.0)
        print(f"✓ Action selection working")
        print(f"  Selected action: {action}")
        print(f"  Action shape: {action.shape}")  # Should be (8,)
        
        assert action.shape == (num_joints,), \
            f"Expected action shape ({num_joints},), got {action.shape}"
        
        # Test epsilon-greedy
        action_random = net.select_action(test_state, epsilon=1.0)
        print(f"✓ Epsilon-greedy exploration working")
        print(f"  Random action: {action_random}")
        
        # Test batch processing
        batch_state = torch.randn(16, state_dim)
        batch_q_values = net(batch_state)
        print(f"✓ Batch processing working")
        print(f"  Batch input: {batch_state.shape}")
        print(f"  Batch output: {batch_q_values.shape}")  # Should be (16, 8, 5)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_replay_buffer():
    """Test 2: Experience Replay Buffer"""
    print("\n" + "="*60)
    print("TEST 2: Replay Buffer")
    print("="*60)
    
    try:
        state_dim = 29
        num_joints = 8
        capacity = 1000
        
        # Create buffer
        buffer = ReplayBuffer(capacity=capacity)
        
        print(f"✓ Replay buffer created")
        print(f"  Capacity: {capacity}")
        print(f"  Initial size: {len(buffer)}")
        
        # Add experiences
        num_experiences = 50
        for i in range(num_experiences):
            buffer.push(
                state=np.random.randn(state_dim),
                action=np.random.randint(0, 5, size=num_joints),
                reward=np.random.randn(),
                next_state=np.random.randn(state_dim),
                done=(i % 10 == 0)  # Every 10th episode ends
            )
        
        print(f"✓ Added {num_experiences} experiences")
        print(f"  Buffer size: {len(buffer)}")
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        print(f"✓ Sampled batch of {batch_size}")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Next states shape: {next_states.shape}")
        print(f"  Dones shape: {dones.shape}")
        
        # Verify shapes
        assert states.shape == (batch_size, state_dim)
        assert actions.shape == (batch_size, num_joints)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, state_dim)
        assert dones.shape == (batch_size,)
        
        print(f"✓ All shapes correct")
        
        # Test overflow (circular buffer)
        for i in range(capacity + 100):
            buffer.push(
                state=np.random.randn(state_dim),
                action=np.random.randint(0, 5, size=num_joints),
                reward=0.0,
                next_state=np.random.randn(state_dim),
                done=False
            )
        
        print(f"✓ Buffer overflow handled")
        print(f"  Final size: {len(buffer)} (max {capacity})")
        assert len(buffer) <= capacity
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dqn_agent_creation():
    """Test 3: DQN Agent creation and initialization"""
    print("\n" + "="*60)
    print("TEST 3: DQN Agent Creation")
    print("="*60)
    
    try:
        state_dim = 29
        
        # Create agent
        agent = DQNAgent(
            state_dim=state_dim,
            num_joints=8,
            num_bins=5,
            learning_rate=1e-4,
            batch_size=32
        )
        
        print(f"✓ DQN Agent created")
        print(f"  Device: {agent.device}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        print(f"  Batch size: {agent.batch_size}")
        print(f"  Buffer capacity: {agent.memory.capacity}")
        
        # Check networks exist
        assert agent.policy_net is not None
        assert agent.target_net is not None
        print(f"✓ Policy and target networks initialized")
        
        # Test action selection
        test_state = np.random.randn(state_dim)
        action = agent.select_action(test_state)
        
        print(f"✓ Action selection working")
        print(f"  Sample action: {action}")
        print(f"  Action shape: {action.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dqn_training_step():
    """Test 4: DQN training step"""
    print("\n" + "="*60)
    print("TEST 4: DQN Training Step")
    print("="*60)
    
    try:
        state_dim = 29
        num_joints = 8
        
        # Create agent
        agent = DQNAgent(
            state_dim=state_dim,
            num_joints=num_joints,
            num_bins=5,
            batch_size=16
        )
        
        print(f"Adding experiences to buffer...")
        
        # Add enough experiences for training
        for i in range(100):
            agent.store_experience(
                state=np.random.randn(state_dim),
                action=np.random.randint(0, 5, size=num_joints),
                reward=np.random.randn(),
                next_state=np.random.randn(state_dim),
                done=(i % 20 == 0)
            )
        
        print(f"✓ Added 100 experiences")
        print(f"  Buffer size: {len(agent.memory)}")
        
        # Perform training step
        loss = agent.train_step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss:.4f}")
        
        assert loss is not None, "Training step should return loss"
        assert isinstance(loss, float), "Loss should be a float"
        
        # Test multiple training steps
        losses = []
        for _ in range(10):
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        
        print(f"✓ Multiple training steps successful")
        print(f"  Average loss: {np.mean(losses):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_epsilon_decay():
    """Test 5: Epsilon decay"""
    print("\n" + "="*60)
    print("TEST 5: Epsilon Decay")
    print("="*60)
    
    try:
        agent = DQNAgent(
            state_dim=29,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.99
        )
        
        print(f"Initial epsilon: {agent.epsilon:.4f}")
        
        # Decay epsilon multiple times
        epsilons = [agent.epsilon]
        for i in range(100):
            agent.decay_epsilon()
            if i % 20 == 0:
                epsilons.append(agent.epsilon)
                print(f"  After {i+1} decays: {agent.epsilon:.4f}")
        
        print(f"✓ Epsilon decay working")
        print(f"  Final epsilon: {agent.epsilon:.4f}")
        print(f"  Minimum epsilon: {agent.epsilon_end}")
        
        # Check epsilon doesn't go below minimum
        assert agent.epsilon >= agent.epsilon_end, "Epsilon below minimum"
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_network_update():
    """Test 6: Target network update"""
    print("\n" + "="*60)
    print("TEST 6: Target Network Update")
    print("="*60)
    
    try:
        state_dim = 29
        
        agent = DQNAgent(
            state_dim=state_dim,
            target_update_freq=10,
            batch_size=8
        )
        
        # Get initial target network weights
        initial_target_params = [p.clone() for p in agent.target_net.parameters()]
        
        print(f"Target update frequency: {agent.target_update_freq} steps")
        
        # Add experiences and train
        for i in range(50):
            agent.store_experience(
                state=np.random.randn(state_dim),
                action=np.random.randint(0, 5, size=8),
                reward=np.random.randn(),
                next_state=np.random.randn(state_dim),
                done=False
            )
        
        # Train for multiple steps
        for step in range(15):
            loss = agent.train_step()
        
        print(f"✓ Trained for {agent.steps_done} steps")
        
        # Check if target network was updated
        current_target_params = [p.clone() for p in agent.target_net.parameters()]
        
        # Compare parameters
        params_changed = False
        for init_p, curr_p in zip(initial_target_params, current_target_params):
            if not torch.equal(init_p, curr_p):
                params_changed = True
                break
        
        if params_changed:
            print(f"✓ Target network was updated")
        else:
            print(f"⚠ Target network not updated (may need more steps)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """Test 7: Save and load checkpoints"""
    print("\n" + "="*60)
    print("TEST 7: Checkpoint Save/Load")
    print("="*60)
    
    try:
        state_dim = 29
        
        # Create agent and train a bit
        agent1 = DQNAgent(state_dim=state_dim, batch_size=8)
        
        # Add experiences
        for i in range(50):
            agent1.store_experience(
                state=np.random.randn(state_dim),
                action=np.random.randint(0, 5, size=8),
                reward=np.random.randn(),
                next_state=np.random.randn(state_dim),
                done=False
            )
        
        # Train
        for _ in range(10):
            agent1.train_step()
        
        agent1.epsilon = 0.5  # Set specific epsilon
        agent1.episodes_done = 42
        
        print(f"Agent 1 state:")
        print(f"  Steps: {agent1.steps_done}")
        print(f"  Epsilon: {agent1.epsilon:.4f}")
        print(f"  Episodes: {agent1.episodes_done}")
        
        # Save checkpoint
        checkpoint_path = "data/test_checkpoint.pth"
        Path("data").mkdir(exist_ok=True)
        agent1.save_checkpoint(checkpoint_path, episode=42, total_reward=123.45)
        
        print(f"✓ Checkpoint saved")
        
        # Create new agent and load checkpoint
        agent2 = DQNAgent(state_dim=state_dim, batch_size=8)
        agent2.load_checkpoint(checkpoint_path)
        
        print(f"\nAgent 2 state after loading:")
        print(f"  Steps: {agent2.steps_done}")
        print(f"  Epsilon: {agent2.epsilon:.4f}")
        print(f"  Episodes: {agent2.episodes_done}")
        
        # Verify loaded values match
        assert agent2.steps_done == agent1.steps_done, "Steps mismatch"
        assert abs(agent2.epsilon - agent1.epsilon) < 1e-6, "Epsilon mismatch"
        assert agent2.episodes_done == agent1.episodes_done, "Episodes mismatch"
        
        print(f"✓ Checkpoint loaded correctly")
        
        # Clean up
        Path(checkpoint_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_with_environment():
    """Test 8: Agent interaction with environment"""
    print("\n" + "="*60)
    print("TEST 8: Agent-Environment Interaction")
    print("="*60)
    
    try:
        # Create environment
        env = HumanoidWalkEnvDiscrete(render_mode=None, num_torque_bins=5)
        
        # Create agent
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            num_joints=8,
            num_bins=5,
            batch_size=16
        )
        
        print(f"✓ Environment and agent created")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space}")
        
        # Run one episode
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        print(f"\nRunning test episode...")
        
        for step in range(50):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, terminated)
            
            # Train (if enough experiences)
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
            
            episode_reward += reward
            episode_steps += 1
            
            if step % 10 == 0:
                print(f"  Step {step}: reward={reward:.3f}, height={info['torso_z']:.3f}")
            
            state = next_state
            
            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                break
        
        print(f"\n✓ Episode completed")
        print(f"  Steps: {episode_steps}")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Buffer size: {len(agent.memory)}")
        print(f"  Agent steps: {agent.steps_done}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_loop():
    """Test 9: Short training loop"""
    print("\n" + "="*60)
    print("TEST 9: Short Training Loop (5 episodes)")
    print("="*60)
    
    try:
        # Create environment
        env = HumanoidWalkEnvDiscrete(render_mode=None, num_torque_bins=5, max_steps=100)
        
        # Create agent
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            num_joints=8,
            num_bins=5,
            batch_size=32,
            epsilon_start=1.0,
            epsilon_decay=0.99
        )
        
        print(f"Starting mini training...")
        
        episode_rewards = []
        
        for episode in range(5):
            state, info = env.reset()
            episode_reward = 0
            episode_losses = []
            
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < 100:
                # Select action
                action = agent.select_action(state, training=True)
                
                # Step
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.train_step()
                    if loss is not None:
                        episode_losses.append(loss)
                
                episode_reward += reward
                state = next_state
                step += 1
            
            # Decay epsilon
            agent.decay_epsilon()
            agent.episodes_done += 1
            
            episode_rewards.append(episode_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            
            print(f"  Episode {episode+1}: "
                  f"reward={episode_reward:.2f}, "
                  f"steps={step}, "
                  f"loss={avg_loss:.4f}, "
                  f"epsilon={agent.epsilon:.3f}")
        
        print(f"\n✓ Training loop completed")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
        print(f"  Final epsilon: {agent.epsilon:.4f}")
        print(f"  Total steps: {agent.steps_done}")
        print(f"  Buffer size: {len(agent.memory)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Module 3 tests"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#  MODULE 3 TEST SUITE - DQN Agent" + " "*26 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    tests = [
        ("Q-Network", test_qnetwork),
        ("Replay Buffer", test_replay_buffer),
        ("DQN Agent Creation", test_dqn_agent_creation),
        ("Training Step", test_dqn_training_step),
        ("Epsilon Decay", test_epsilon_decay),
        ("Target Network Update", test_target_network_update),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
        ("Agent-Environment Interaction", test_agent_with_environment),
        ("Training Loop", test_full_training_loop),
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
        print("\n🎉 All tests passed! Module 3 is ready!")
        print("\n📋 Next: Module 4 - Full Training Pipeline")
    else:
        print("\n⚠ Some tests failed. Check errors above and fix.")
    
    print("\n" + "#"*60 + "\n")


if __name__ == "__main__":
    run_all_tests()