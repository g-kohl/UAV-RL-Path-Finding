import argparse
from environment import Environment
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument('--timesteps', type=int, required=True)
parser.add_argument('--grid_size', nargs='+', required=True)
parser.add_argument('--static_obstacles', action='store_true', default=False)
parser.add_argument('--mobile_obstacles', action='store_true', default=False)

arguments = parser.parse_args()

timesteps = arguments.timesteps
grid_size = (int(arguments.grid_size[0]), int(arguments.grid_size[1]))
static_obstacles = arguments.static_obstacles
mobile_obstacles = arguments.mobile_obstacles

environment = Environment(grid_size=grid_size, static_obstacles=static_obstacles, mobile_obstacles=mobile_obstacles)
check_env(environment, warn=True)

evaluate_environment = Environment(seed=42)

evaluate_callback = EvalCallback(
    evaluate_environment,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=100000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

if os.path.exists("models/dqn_uav.zip"):
    print("Model found. Continuing training...")
    model = DQN.load("models/dqn_uav", env=environment)
    reset_timesteps = False
else:
    print("No model found. New training starting...")
    model = DQN(
        policy="MlpPolicy",
        env=environment,
        verbose=1,
        learning_rate=3e-5,
        buffer_size=200000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        train_freq=16,
        target_update_interval=5000,
        gradient_steps=1
    )

    reset_timesteps = True

model.learn(total_timesteps=timesteps,
            callback=evaluate_callback,
            reset_num_timesteps=reset_timesteps,
            tb_log_name="DQN_training")

model.save("models/dqn_uav")
