from stable_baselines3 import DQN
from environment import Environment
import time

environment = Environment()
model = DQN.load("models/dqn_uav", env=environment)

observation = environment.reset()
done = False

while not done:
    action, _ = model.predict(observation)
    observation, reward, done, info = environment.step(action)
    environment.render()
    time.sleep(0.5)
