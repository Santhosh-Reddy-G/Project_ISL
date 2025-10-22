"""
Main Entry Point for Humanoid Locomotion DQN Training
Complete end-to-end pipeline from pose extraction to trained walking agent
"""

import argparse
import sys
from pathlib import Path

# Import training functions
from training.trainer import train_agent, evaluate_trained_agent


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Deep RL for Humanoid Locomotion from Image-Defined Initial Pose',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train agent for 5000 episodes
  python main.py --mode train --episodes 5000
  
  # Quick test training (100 episodes)
  python main.py --mode train --episodes 100 --name quick_test
  
  # Evaluate trained agent
  python main.py --mode eval --checkpoint models/checkpoints/best_model.pth
  
  # Watch agent walk (with GUI)
  python main.py --mode demo --checkpoint models/checkpoints/best_model.pth --render
  
  # Generate pose library from images
  python main.py --mode generate_poses --image_folder data/input_images
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'eval', 'demo', 'generate_poses'],
        help='Operation mode'
    )
    
    # Training arguments
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of training episodes (default: 5000)')
    parser.add_argument('--name', type=str, default='default',
                       help='Experiment name for checkpoints (default: default)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Epsilon decay rate (default: 0.995)')
    parser.add_argument('--no_pose_lib', action='store_true',
                       help='Disable pose library (use default standing pose)')
    
    # Evaluation arguments
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint for evaluation/demo')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes (for demo mode)')
    
    # Pose generation arguments
    parser.add_argument('--image_folder', type=str, default='data/input_images',
                       help='Folder containing input images')
    parser.add_argument('--output_folder', type=str, default='data/pose_library',
                       help='Output folder for pose library')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'train':
        print("\n" + "="*60)
        print("MODE: TRAINING")
        print("="*60 + "\n")
        
        # Configuration
        env_config = {
            'render_mode': None,
            'max_steps': 1000,
            'num_torque_bins': 5
        }
        
        agent_config = {
            'hidden_dims': [256, 256, 128],
            'learning_rate': args.lr,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': args.epsilon_decay,
            'buffer_capacity': 100000,
            'batch_size': args.batch_size,
            'target_update_freq': 1000
        }
        
        training_config = {
            'num_episodes': args.episodes,
            'eval_freq': 50,
            'eval_episodes': 10,
            'save_freq': 100,
            'log_freq': 10,
            'checkpoint_dir': f'models/{args.name}',
            'use_pose_library': not args.no_pose_lib,
            'pose_library_path': 'data/pose_library',
            'early_stopping': True,
            'patience': 200,
            'min_reward_threshold': 50.0
        }
        
        print(f"Training Configuration:")
        print(f"  Episodes: {args.episodes}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epsilon decay: {args.epsilon_decay}")
        print(f"  Pose library: {'Enabled' if not args.no_pose_lib else 'Disabled'}")
        print(f"  Checkpoint dir: {training_config['checkpoint_dir']}")
        print()
        
        # Train
        train_agent(
            env_config=env_config,
            agent_config=agent_config,
            training_config=training_config
        )
    
    elif args.mode == 'eval':
        print("\n" + "="*60)
        print("MODE: EVALUATION")
        print("="*60 + "\n")
        
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation mode")
            sys.exit(1)
        
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        evaluate_trained_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.eval_episodes,
            render=False
        )
    
    elif args.mode == 'demo':
        print("\n" + "="*60)
        print("MODE: DEMONSTRATION")
        print("="*60 + "\n")
        
        if not args.checkpoint:
            print("Error: --checkpoint required for demo mode")
            sys.exit(1)
        
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        print("Starting demonstration (GUI will open)...")
        print("Press Ctrl+C to stop\n")
        
        evaluate_trained_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.eval_episodes,
            render=True
        )
    
    elif args.mode == 'generate_poses':
        print("\n" + "="*60)
        print("MODE: GENERATE POSE LIBRARY")
        print("="*60 + "\n")
        
        from perception.pose_pipeline import generate_pose_library
        
        if not Path(args.image_folder).exists():
            print(f"Error: Image folder not found: {args.image_folder}")
            sys.exit(1)
        
        print(f"Generating pose library from: {args.image_folder}")
        print(f"Output folder: {args.output_folder}\n")
        
        pose_library = generate_pose_library(
            image_folder=args.image_folder,
            output_folder=args.output_folder,
            visualize=True
        )
        
        print(f"\n✓ Generated {len(pose_library)} poses")
        print(f"✓ Saved to: {args.output_folder}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)