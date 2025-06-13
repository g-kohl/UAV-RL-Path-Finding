import numpy as np
import matplotlib.pyplot as plt

data = np.load("./logs/evaluations.npz")

timesteps = data["timesteps"]
results = data["results"]

mean_rewards = results.mean(axis=1)

plt.figure(figsize=(10, 5))
plt.plot(timesteps, mean_rewards)
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Training Progress")
plt.grid()
plt.show()