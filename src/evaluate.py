import argparse
import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3 # Import registered algorithms
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from utils import make_env # Import make_env for creating single environments

# Define available algorithms (must match train.py or be auto-detected by SB3)
ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "sac": SAC,
    "td3": TD3,
}

def evaluate_agent(args):
    """
    Evaluates a trained Stable Baselines3 agent on multiple highway-env environments.
    """
    print(f"--- Starting Evaluation ---")
    print(f"Model Path: {args.model_path}")
    print(f"Environments: {args.envs}")
    print(f"Evaluation Episodes per Env: {args.n_eval_episodes}")
    print(f"Seed: {args.seed}")
    print(f"Render: {args.render}")
    print(f"Record Video: {args.record_video}")
    if args.record_video:
        print(f"Video Directory: {args.video_dir}")
    print(f"-------------------------")

    # --- Load Model ---
    # Extract algorithm name from model path if possible, otherwise require --algo flag
    # This is a simple heuristic, might need adjustment based on naming convention
    model_filename = os.path.basename(args.model_path)
    algo_name_match = None
    for name in ALGOS.keys():
        # Check if algo name is part of the filename (e.g., highway-v0_ppo_final.zip)
        if f"_{name}_" in model_filename.lower() or model_filename.lower().startswith(name + "_"):
             algo_name_match = name
             break

    if algo_name_match:
        print(f"Inferred algorithm: {algo_name_match.upper()}")
        AlgoClass = ALGOS[algo_name_match]
    elif args.algo:
         print(f"Using specified algorithm: {args.algo.upper()}")
         if args.algo.lower() not in ALGOS:
              raise ValueError(f"Specified algorithm '{args.algo}' not found.")
         AlgoClass = ALGOS[args.algo.lower()]
    else:
        raise ValueError("Could not infer algorithm from model filename. Please specify the algorithm using the --algo flag.")


    try:
        model = AlgoClass.load(args.model_path)
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []
    # --- Evaluate on each environment ---
    for env_id in args.envs:
        print(f"\nEvaluating on environment: {env_id}")

        # Create the evaluation environment
        # Use a single environment for evaluation and rendering/recording
        eval_env_fns = [make_env(env_id, args.seed, rank=i, log_dir=None, env_kwargs=args.env_config) for i in range(args.n_render_envs if (args.render or args.record_video) else 1)]
        eval_env = DummyVecEnv(eval_env_fns)


        # --- Video Recording Setup ---
        if args.record_video:
            video_folder = os.path.join(args.video_dir, env_id, time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(video_folder, exist_ok=True)
            print(f"Recording video to: {video_folder}")
            eval_env = VecVideoRecorder(eval_env, video_folder,
                                        record_video_trigger=lambda x: x == 0, # Record the first episode
                                        video_length=args.video_length, # Max length of recorded video
                                        name_prefix=f"{algo_name_match or args.algo}-{env_id}")

        # --- Run Evaluation Episodes ---
        episode_rewards = []
        episode_lengths = []
        successes = [] # Track success rate if available

        for episode in range(args.n_eval_episodes):
            obs, _ = eval_env.reset()
            done = np.zeros(eval_env.num_envs, dtype=bool) # Track done state for each env if VecEnv is used
            ep_reward = np.zeros(eval_env.num_envs)
            ep_len = np.zeros(eval_env.num_envs, dtype=int)
            ep_success = np.zeros(eval_env.num_envs, dtype=bool)

            while not np.all(done):
                action, _states = model.predict(obs, deterministic=True)
                new_obs, reward, terminated, truncated, infos = eval_env.step(action)

                # Handle rendering if enabled (only practical for n_render_envs=1)
                if args.render and eval_env.num_envs == 1:
                    try:
                        eval_env.render()
                        # Add a small delay to make rendering visible
                        time.sleep(0.02)
                    except Exception as e:
                        print(f"Warning: Rendering failed - {e}")
                        args.render = False # Disable rendering if it fails

                # Update episode stats for environments that are not yet done
                ep_reward[~done] += reward[~done]
                ep_len[~done] += 1

                # Check for success (adapt based on specific env's info dict)
                # Common patterns: info['is_success'], info['rewards']['success']
                # Fallback: Check if terminated (goal reached) and not truncated (time limit)
                for i, info in enumerate(infos):
                     if not done[i]: # Only update if the episode for this env isn't done yet
                        is_success = info.get("is_success", False)
                        # Alternative check if 'is_success' not present
                        if not is_success and terminated[i] and not truncated[i]:
                             is_success = True # Assume termination without truncation means success
                        ep_success[i] = is_success if terminated[i] or truncated[i] else ep_success[i] # Store success only at episode end


                # Update done status for all environments
                # Important: Use the combined terminated or truncated signal
                current_done = terminated | truncated
                done = current_done

                obs = new_obs # Update observation

            # Append results from the completed episode(s)
            episode_rewards.extend(ep_reward.tolist())
            episode_lengths.extend(ep_len.tolist())
            successes.extend(ep_success.tolist())

            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{args.n_eval_episodes} completed.")

        # Close the environment (especially important for video recorder)
        eval_env.close()

        # --- Calculate Statistics ---
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_rate = np.mean(successes) if successes else 0.0 # Handle case where success info wasn't available

        print(f"  Results for {env_id}:")
        print(f"    Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"    Mean Episode Length: {mean_length:.2f}")
        print(f"    Success Rate: {success_rate:.2%}")

        results.append({
            "Environment": env_id,
            "Mean Reward": mean_reward,
            "Std Reward": std_reward,
            "Mean Length": mean_length,
            "Success Rate": success_rate,
            "Episodes": args.n_eval_episodes
        })

    # --- Output Summary Table ---
    results_df = pd.DataFrame(results)
    print("\n--- Evaluation Summary ---")
    print(results_df.to_string(index=False))

    # Optionally save results to CSV
    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nEvaluation results saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent on highway-env")

    # Model and Environment(s)
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained agent (.zip file)")
    parser.add_argument("--envs", nargs='+', required=True, help="List of Gym environment IDs to evaluate on")
    parser.add_argument("--algo", type=str, default=None, choices=list(ALGOS.keys()), help="RL Algorithm (if it cannot be inferred from model path)")

    # Evaluation Parameters
    parser.add_argument("--n-eval-episodes", type=int, default=20, help="Number of episodes to run for evaluation per environment")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for evaluation")

    # Visualization and Recording
    parser.add_argument("--render", action='store_true', help="Render the environment during evaluation (only works well with --n-render-envs 1)")
    parser.add_argument("--n-render-envs", type=int, default=1, help="Number of parallel environments for rendering/recording (usually 1)")
    parser.add_argument("--record-video", action='store_true', help="Record videos of the evaluation runs")
    parser.add_argument("--video-dir", type=str, default="./videos/", help="Directory to save recorded videos")
    parser.add_argument("--video-length", type=int, default=500, help="Maximum length of recorded videos (in steps)")

    # Output
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save the evaluation results as a CSV file (optional)")

    # Environment Configuration (Example: pass as JSON string)
    parser.add_argument("--env-config", type=str, default='{}', help="JSON string for environment configuration kwargs")


    args = parser.parse_args()

    # Parse environment config from JSON string
    import json
    try:
        args.env_config = json.loads(args.env_config)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse --env-config JSON string: {args.env_config}. Using default env config.")
        args.env_config = {}

    if args.render and args.n_render_envs > 1:
        print("Warning: Rendering with more than one environment (--n-render-envs > 1) may not display correctly. Setting n_render_envs=1 for rendering.")
        args.n_render_envs = 1

    evaluate_agent(args)


