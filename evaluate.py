from stable_baselines3 import DQN
from environment import Environment
import time

environment = Environment(grid_size=(10, 10), render_mode="human")
model = DQN.load("models/best_model", env=environment)

for _ in range(10):
    observation, _ = environment.reset()
    terminated = False

    while not terminated:
        action, _ = model.predict(observation)
        observation, _, terminated, _, _ = environment.step(action)
        environment.render()
        time.sleep(1.0)
