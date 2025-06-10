from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from environment import Environment

environment = Environment()

check_env(environment, warn=True)

model = DQN(
    policy="MlpPolicy",
    env=environment,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.02,
    train_freq=4,
    target_update_interval=100
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="models/",
    name_prefix="dqn_uav"
)

model.learn(total_timesteps=1000000, callback=checkpoint_callback)
model.save("models/dqn_uav")
