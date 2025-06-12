from stable_baselines3 import DQN
from environment import Environment
import time

environment = Environment(render_mode="human")
model = DQN.load("models/dqn_uav", env=environment)

for _ in range(10):
    observation, _ = environment.reset()
    terminated = False

    while not terminated:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = environment.step(action)
        environment.render()
        time.sleep(1.0)
