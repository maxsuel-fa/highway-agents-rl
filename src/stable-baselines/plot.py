import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results


def plot(data, title, what='Reward', window_size=100):
    data_smooth = data.rolling(window_size, min_periods=1).mean()
    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot raw rewards (optional, can be noisy)
    ax.plot(data, color='lightblue', alpha=0.5, label=f'Episode {what} (Raw)')

    # Plot smoothed rewards
    ax.plot(data_smooth, color='blue', label=f'Episode {what} (Smoothed, w={window_size})')

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Episode {what}")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Improve x-axis formatting (optional)
    """try:
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    except ImportError:
        print("Note: Install matplotlib for better plot formatting if needed.")
    """

    plt.tight_layout()
    plt.show()


def plot_learning_curves(log_folder: str, title: str = "Learning Curve", output_file: str | None = None):
    """
    Plots the learning curve from Stable Baselines3 Monitor logs.

    :param log_folder: Path to the directory containing the monitor.csv file(s).
                       If multiple monitor files exist (from VecMonitor), they will be loaded.
    :param title: Title for the plot.
    :param output_file: Path to save the plot image (e.g., 'learning_curve.png'). If None, displays the plot.
    """
    # Load results using the SB3 helper function
    # This automatically handles potentially multiple monitor files in the folder
    monitor_data = load_results(log_folder)
    print(monitor_data)
    if len(monitor_data) == 0:
        print(f"Warning: No monitor files found or readable in {log_folder}")
        return

    # Extract timestamps, episode lengths, and rewards
    # 't' is wall-clock time, 'l' is episode length, 'r' is episode reward
    # We need cumulative timesteps, which SB3 adds as 'total_timesteps' when loading
    #timesteps = monitor_data["t"]
    rewards = monitor_data["r"] # Episode rewards
    durations = monitor_data['l']

    # Calculate rolling mean for smoother curve (optional)
    window_size = 100 # Adjust window size as needed
    plot(rewards, 'Training Reward')
    plot(durations, 'Training Episodes Length', 'Episode Length')

    """except LoadMonitorResultsError:
        print(f"Error: Could not load monitor results from {log_folder}. Ensure 'monitor.csv' exists.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from Stable Baselines3 Monitor logs.")
    parser.add_argument("--log-dir", required=True, type=str, help="Directory containing the monitor.csv file(s). This should be the specific run directory (e.g., ./logs/highway-v0_PPO_20231027_100000).")
    parser.add_argument("--title", type=str, default="Learning Curve", help="Title for the plot.")
    parser.add_argument("--output-file", type=str, default=None, help="Path to save the plot image (e.g., ./learning_curve.png). If not specified, displays the plot.")

    args = parser.parse_args()

    # Try to infer a title if not provided explicitly
    if args.title == "Learning Curve":
        # Extract env_id and algo from the log directory path if possible
        try:
            parts = os.path.basename(os.path.normpath(args.log_dir)).split('_')
            if len(parts) >= 2:
                 # Simple heuristic, adjust if your naming convention differs
                env_id_part = parts[0]
                algo_part = parts[1]
                args.title = f"Learning Curve: {env_id_part} ({algo_part})"
        except:
            pass # Keep default title if parsing fails

    plot_learning_curves(args.log_dir, args.title, args.output_file)

