import argparse
from environment import Environment
from stable_baselines3 import DQN, PPO
import time

INTERVAL = 0.5

parser = argparse.ArgumentParser()

parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--grid_size', nargs='+', default=[15, 15])
parser.add_argument('--static_obstacles', action='store_true', default=False)
parser.add_argument('--mobile_obstacles', action='store_true', default=False)
parser.add_argument('--model', type=str, default="models/last_model")
parser.add_argument('--algorithm', type=str, default="PPO")

arguments = parser.parse_args()

episodes = arguments.episodes

if episodes > 10:
    step_time = 0
    render_mode = None
    render = False
else:
    step_time = INTERVAL
    render_mode = "human"
    render = True

grid_size = (int(arguments.grid_size[0]), int(arguments.grid_size[1]))
static_obstacles = arguments.static_obstacles
mobile_obstacles = arguments.mobile_obstacles
model_name = arguments.model
algorithm = arguments.algorithm

environment = Environment(grid_size=grid_size,
                          static_obstacles=static_obstacles,
                          mobile_obstacles=mobile_obstacles,
                          training=False,
                          seed=42,
                          render_mode=render_mode)

if algorithm == "DQN":
    model = DQN.load(model_name, env=environment)
elif algorithm == "PPO":
    model = PPO.load(model_name, env=environment)

total_reward = 0
successes = 0
collisions = 0
truncations = 0

print(f"Evaluating {model_name}")

for episode in range(episodes):
    observation, _ = environment.reset()
    terminated = truncated = False
    episode_reward = 0

    while not (terminated or truncated):
        if render:
            environment.render()
        
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, _ = environment.step(action)
        episode_reward += reward

        if terminated and reward > 0:
            successes += 1
        elif terminated and reward < 0:
            collisions += 1
        elif truncated:
            truncations += 1

        time.sleep(step_time)

    if render:
        print(f"Episode {episode} reward: {episode_reward}")

    total_reward += episode_reward

print(f"Successes: {successes}")
print(f"Collisions: {collisions}")
print(f"Truncations: {truncations}")
print(f"Mean reward: {(total_reward / episodes):.3f}")
