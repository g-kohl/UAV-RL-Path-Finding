import argparse
from environment import Environment
from stable_baselines3 import DQN
import time

parser = argparse.ArgumentParser()

parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--grid_size', nargs='+', default=[15, 15])
parser.add_argument('--static_obstacles', action='store_true', default=False)
parser.add_argument('--mobile_obstacles', action='store_true', default=False)

arguments = parser.parse_args()

episodes = arguments.episodes
grid_size = (int(arguments.grid_size[0]), int(arguments.grid_size[1]))
static_obstacles = arguments.static_obstacles
mobile_obstacles = arguments.mobile_obstacles

environment = Environment(grid_size=grid_size, static_obstacles=static_obstacles, mobile_obstacles=mobile_obstacles, render_mode="human")
model = DQN.load("models/best_model", env=environment)

total_reward = 0

for episode in range(episodes):
    observation, _ = environment.reset()
    terminated = truncated = False
    episode_reward = 0

    while not (terminated or truncated):
        environment.render()
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, _ = environment.step(action)
        episode_reward += reward
        time.sleep(0.75)

    print(f"Episode {episode} reward: {episode_reward}")
    total_reward += episode_reward

print(f"Mean reward: {(total_reward / episodes):.3f}")
