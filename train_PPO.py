import argparse
from environment import Environment
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
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

checkpoint_callback = CheckpointCallback(
    save_freq=1_000_000,
    save_path="./models/checkpoints/",
    name_prefix="model"
)

callbacks = CallbackList([evaluate_callback, checkpoint_callback])

if os.path.exists("models/pretrained_model.zip"):
    print("Pre-trained model found. Adaptation training starting...")

    model = PPO.load("models/pretrained_model", env=environment)
    model.learning_rate = 1e-5
    reset_timesteps = False

elif os.path.exists("models/last_model.zip"):
    print("Model found. Continuing training...")

    model = PPO.load("models/last_model", env=environment)
    reset_timesteps = False

else:
    print("No model found. New training starting...")

    model = PPO(
        policy="MlpPolicy",
        env=environment,
        verbose=1,
        learning_rate=3e-5,
        gamma=0.995,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
    )

    reset_timesteps = True

model.learn(total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_timesteps,
            tb_log_name="PPO_training")

model.save("models/last_model")
