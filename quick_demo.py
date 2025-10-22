import time
from control.dqn_agent import DQNAgent
from simulation.humanoid_env import HumanoidWalkEnvDiscrete

# Load environment and agent
env = HumanoidWalkEnvDiscrete(render_mode='human', max_steps=500)
agent = DQNAgent(state_dim=29, num_joints=8, num_bins=5)
agent.load_checkpoint('models/full_training/best_model.pth')

print("Starting demo... Watch the PyBullet GUI!")

for episode in range(10):
    print(f"\n=== Episode {episode + 1} ===")
    state, info = env.reset()
    done = False
    step = 0
    
    while not done and step < 500:
        action = agent.select_action(state, training=False)
        state, reward, done, truncated, info = env.step(action)
        
        time.sleep(0.01)  # Slow down (30ms per step)
        step += 1
        
        if step % 50 == 0:
            print(f"  Step {step}: x={info['torso_x']:.2f}m, z={info['torso_z']:.2f}m")
    
    print(f"  Final distance: {info['torso_x']:.2f}m")
    time.sleep(2)  # Pause between episodes

env.close()