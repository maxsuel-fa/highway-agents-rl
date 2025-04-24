import os
import pickle


config_dict = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "ContinuousAction"
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "other_vehicles": 4
}

curr_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(curr_dir, 'parking_config.pkl')
with open(config_path, "wb") as f:
    pickle.dump(config_dict, f)

