from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from environment import Environment

environment = Environment()

check_env(environment, warn=True)

model = DQN(
    policy="MultiInputPolicy",
    env=environment,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    tau=0.05,
    train_freq=4,
    target_update_interval=100
)

model.learn(total_timesteps=50000)
model.save("models/dqn_uav")
