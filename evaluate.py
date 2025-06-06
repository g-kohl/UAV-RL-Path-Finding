from stable_baselines3 import DQN
from environment import Environment

environment = Environment()
model = DQN.load("saved_models/dqn_drone", env=environment)

observation = environment.reset()
done = False

while not done:
    action, _ = model.predict(observation)
    observation, reward, done, info = environment.step(action)
    environment.render()
