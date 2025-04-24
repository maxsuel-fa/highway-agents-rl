import argparse
import os
import time
from datetime import datetime
import gymnasium as gym

from stable_baselines3 import PPO, A2C, DQN, SAC, TD3 # Import other algorithms as needed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from utils import make_env, create_vectorized_env, SaveCheckpointCallback # Import from utils.py

# Define available algorithms
ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "sac": SAC,
    "td3": TD3,
}

def train_agent(args):
    """
    Trains a Stable Baselines3 agent on a highway-env environment.
    """
    # --- Setup ---
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_id}_{args.algo}_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    model_save_path = os.path.join(args.model_save_dir, f"{args.env_id}_{args.algo}") # Base path for final model
    checkpoint_save_path = os.path.join(args.model_save_dir, "checkpoints", run_name) # Path for intermediate checkpoints
    tensorboard_log_path = os.path.join(log_dir, "tensorboard")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True) # Ensure base model dir exists
    os.makedirs(checkpoint_save_path, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)

    print(f"--- Starting Training ---")
    print(f"Environment: {args.env_id}")
    print(f"Algorithm: {args.algo}")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Log Directory: {log_dir}")
    print(f"Model Save Directory: {args.model_save_dir}")
    print(f"Using {args.n_envs} parallel environments.")
    print(f"Seed: {args.seed}")
    print(f'{tensorboard_log_path}')
    print(f"-------------------------")

    # --- Environment Creation ---
    # Create the vectorized training environment
    # Pass Monitor logs to the main log_dir for VecMonitor aggregation
    train_env = create_vectorized_env(
        env_id=args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        log_dir=log_dir, # VecMonitor will save aggregated stats here
        env_kwargs=args.env_config # Pass environment configuration
    )

    # Create a separate environment for evaluation (recommended practice)
    # Use DummyVecEnv as evaluation is usually done sequentially
    eval_env = create_vectorized_env(
        env_id=args.env_id,
        n_envs=1, # Single env for evaluation
        seed=args.seed + 1000, # Use a different seed for eval env
        log_dir=None, # No Monitor logging needed for eval callback env usually
        env_kwargs=args.env_config
    )
    # If using VecNormalize, wrap the eval env and sync stats periodically
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)


    # --- Callbacks ---
    # Tensorboard callback
    #tensorboard_callback = TensorBoardCallback(tensorboard_log_path)

    # Evaluation callback: Runs evaluation periodically and saves the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_path, "best_model"), # Save best model here
        log_path=log_dir, # Logs evaluation results here
        eval_freq=max(args.eval_freq // args.n_envs, 1), # Adjust frequency for vec envs
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Custom Checkpoint callback: Saves model checkpoints periodically
    checkpoint_callback = SaveCheckpointCallback(
        check_freq=max(args.checkpoint_freq // args.n_envs, 1), # Adjust frequency
        save_path=checkpoint_save_path,
        name_prefix=f"{args.env_id}_{args.algo}",
        verbose=1
    )

    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # --- Agent Initialization ---
    if args.algo.lower() not in ALGOS:
        raise ValueError(f"Algorithm '{args.algo}' not supported. Choose from: {list(ALGOS.keys())}")

    AlgoClass = ALGOS[args.algo.lower()]

    # Define policy kwargs based on algorithm if needed (e.g., for DQN)
    policy_kwargs = {}
    if args.algo.lower() == "dqn":
        # Example: Configure DQN network architecture
        # policy_kwargs = dict(net_arch=[256, 256])
        pass # Add specific DQN policy kwargs if needed

    # Define algorithm hyperparameters
    # These could be loaded from a YAML file or passed via argparse
    algo_kwargs = dict(
        policy=args.policy,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tensorboard_log_path, # Redundant if using TensorBoardCallback, but doesn't hurt
        device='cpu'
    )

    # Filter kwargs for the specific algorithm (e.g., DQN doesn't use n_steps, gae_lambda etc.)
    # This requires knowing the specific args for each algo, or using a more robust config system
    if args.algo.lower() == "dqn":
        dqn_keys = ['policy', 'learning_rate', 'buffer_size', 'learning_starts',
                    'batch_size', 'tau', 'gamma', 'train_freq', 'gradient_steps',
                    'replay_buffer_class', 'replay_buffer_kwargs', 'optimize_memory_usage',
                    'target_update_interval', 'exploration_fraction', 'exploration_initial_eps',
                    'exploration_final_eps', 'max_grad_norm', 'seed', 'verbose', 'tensorboard_log']
        algo_kwargs = {k: v for k, v in algo_kwargs.items() if k in dqn_keys}
        # Add default DQN specific args if not provided
        algo_kwargs.setdefault('buffer_size', 100_000)
        algo_kwargs.setdefault('learning_starts', 1000)
        algo_kwargs.setdefault('target_update_interval', 1000)
        algo_kwargs.setdefault('exploration_fraction', 0.1)
        algo_kwargs.setdefault('exploration_final_eps', 0.05)


    model = AlgoClass(
        env=train_env,
        policy_kwargs=policy_kwargs,
        **algo_kwargs
    )

    # --- Training ---
    print(f"Starting training for {args.total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            log_interval=args.log_interval, # Log training stats every N updates
            tb_log_name=run_name,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # --- Save Final Model ---
        final_model_path = os.path.join(model_save_path, f"{args.env_id}_{args.algo}_final.zip")
        print(f"Saving final model to {final_model_path}")
        model.save(final_model_path)

        # Close environments
        train_env.close()
        eval_env.close()

        end_time = time.time()
        print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes.")
        print(f"Final model saved to: {final_model_path}")
        print(f"TensorBoard logs saved to: {tensorboard_log_path}")
        print(f"You can view logs by running: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on highway-env")

    # Environment and Algorithm
    parser.add_argument("--env-id", type=str, default="highway-v0", help="Gym environment ID")
    parser.add_argument("--algo", type=str, default="PPO", choices=list(ALGOS.keys()), help="RL Algorithm")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy network type (e.g., MlpPolicy, CnnPolicy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")

    # Training Duration and Logging
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total number of training timesteps")
    parser.add_argument("--log-dir", type=str, default="./logs/", help="Directory to save logs")
    parser.add_argument("--model-save-dir", type=str, default="./models/", help="Directory to save trained models")
    parser.add_argument("--log-interval", type=int, default=10, help="Log training stats every N updates (1 update = n_steps * n_envs timesteps)")

    # Callbacks
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluate the agent every N timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--checkpoint-freq", type=int, default=20000, help="Save a model checkpoint every N timesteps")

    # Algorithm Hyperparameters (Common for PPO)
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps to run for each environment per update (PPO)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (PPO, DQN)")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of optimization epochs per update (PPO)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Factor for trade-off of bias vs variance for GAE (PPO)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Clipping parameter (PPO)")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max value for gradient clipping")

    # Environment Configuration (Example: pass as JSON string)
    # e.g., --env-config '{"observation": {"type": "Kinematics"}, "action": {"type": "DiscreteMetaAction"}}'
    parser.add_argument("--env-config", type=str, default='{}', help="JSON string for environment configuration kwargs")

    args = parser.parse_args()

    # Parse environment config from JSON string
    import json
    try:
        args.env_config = json.loads(args.env_config)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse --env-config JSON string: {args.env_config}. Using default env config.")
        args.env_config = {}


    train_agent(args)

