import os
import gymnasium as gym
import highway_env # Import the package - this should handle registration
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

# REMOVED: highway_env.register_highway_envs() - Registration usually happens on import

def make_env(env_id: str, seed: int, rank: int = 0, log_dir: str | None = None, env_kwargs: dict | None = None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID (e.g., "highway-v0")
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess or environment instance
    :param log_dir: directory for Monitor wrapper logging. If None, Monitor is not used.
    :param env_kwargs: Optional keyword arguments to pass to the env constructor
    :return: a function that creates the environment
    """
    def _init():
        # Ensure env_kwargs is a dictionary if None is passed
        env_actual_kwargs = env_kwargs if env_kwargs is not None else {}
        # Create the environment
        env = gym.make(env_id, **env_actual_kwargs)
        # Seed the environment
        # Important: use a different seed for each environment instance (seed + rank)
        # Note: gymnasium's reset now handles seeding. Initial seeding might still be useful
        # depending on the environment's implementation details, but reset is primary.
        # env.seed(seed + rank) # Deprecated in Gymnasium
        # env.action_space.seed(seed + rank) # Also potentially needed for reproducibility

        # The primary way to seed in Gymnasium is via reset
        # However, we call reset outside this function now, when the VecEnv is created or reset.
        # We might still need to set the seed attribute if the env uses it internally outside reset
        # setattr(env.unwrapped, 'seed', seed + rank) # Less common now

        if log_dir is not None:
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            # Define path for the Monitor log file for this specific environment instance
            log_path = os.path.join(log_dir, str(rank))
            # Wrap the environment with the Monitor wrapper to log episode statistics
            env = Monitor(env, log_path)

        # The environment is reset with the seed when the VecEnv calls reset
        # env.reset(seed=seed + rank) # Don't reset here, VecEnv handles it

        return env
    # Set the seed for the init function itself if needed (less common)
    # seed_fn(seed)
    return _init

def create_vectorized_env(env_id: str, n_envs: int, seed: int, log_dir: str | None, env_kwargs: dict | None = None):
    """
    Creates a vectorized (potentially parallel) environment.

    :param env_id: The environment ID.
    :param n_envs: Number of parallel environments.
    :param seed: Base seed for environments.
    :param log_dir: Directory for Monitor logs. If None, monitoring is disabled.
    :param env_kwargs: Optional keyword arguments for the environment.
    :return: A Stable Baselines3 VecEnv.
    """
    if log_dir:
        # Ensure the base log directory exists
        os.makedirs(log_dir, exist_ok=True)

    # Create a list of environment thunks (functions that create the env)
    env_fns = [make_env(env_id, seed, i, log_dir, env_kwargs) for i in range(n_envs)]

    # Create the vectorized environment
    if n_envs > 1:
        # Use SubprocVecEnv for multiprocessing (runs each env in a separate process)
        # This is generally faster for complex environments but has overhead.
        vec_env = SubprocVecEnv(env_fns, start_method='fork') # 'fork' is often faster on Linux/macOS
    else:
        # Use DummyVecEnv for a single environment or when multiprocessing is not desired
        # Runs environments sequentially in the main process.
        vec_env = DummyVecEnv(env_fns)

    # Wrap with VecMonitor to aggregate statistics from individual Monitor wrappers
    # This reads the .monitor.csv files created by the Monitor wrapper in each sub-process/dummy env
    # and logs aggregated stats (mean reward, length etc.) to a single file.
    if log_dir is not None:
        monitor_log_path = os.path.join(log_dir, "monitor.csv")
        # Automatically logs mean reward, length etc. to monitor.csv and console
        vec_env = VecMonitor(vec_env, monitor_log_path)

    # Set the random seeds for the vectorized environment.
    # This calls reset() on each environment with the appropriate seed.
    vec_env.seed(seed)
    # It's also good practice to seed the action space if it's stochastic
    vec_env.action_space.seed(seed)

    return vec_env


class SaveCheckpointCallback(BaseCallback):
    """
    Callback for saving a model checkpoint periodically during training.

    Saves checkpoints based on the total number of timesteps observed by the agent.

    :param check_freq: Frequency (in total timesteps) at which to save the model.
                       This frequency is relative to the total timesteps, not per environment.
    :param save_path: Path to the directory where checkpoints will be saved.
    :param name_prefix: Prefix for the checkpoint file name. Defaults to "rl_model".
    :param verbose: Verbosity level (0 for no output, 1 for info messages, 2 for debug).
    """
    def __init__(self, check_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        # Ensure the save path exists when the callback is initialized
        os.makedirs(self.save_path, exist_ok=True)

    def _init_callback(self) -> None:
        """Called before training starts."""
        # Redundant check, but ensures path exists if created between init and training start
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment (across all parallel envs).
        `self.num_timesteps` holds the total number of timesteps encountered so far.
        `self.n_calls` holds the number of times this callback method has been called (usually equals `self.num_timesteps`).

        :return: bool: If False, training is aborted. True otherwise.
        """
        # Check if the current total number of timesteps is a multiple of the check frequency
        # Using self.num_timesteps which is the total steps environment steps collected so far
        if self.num_timesteps % self.check_freq == 0:
            # Construct the full path for the checkpoint file
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            # Save the current state of the model
            self.model.save(path)
            if self.verbose > 0: # Print only if verbose level is 1 or higher
                print(f"Saving model checkpoint to {path} (timestep {self.num_timesteps})")
        # Return True to continue training
        return True

