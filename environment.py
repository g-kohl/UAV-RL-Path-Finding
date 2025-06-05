import gym
from gym import spaces
import numpy as np

class Environment(gym.Env):
    def __init__(self):
        super(Environment, self).__init__()

        self.grid_size = np.array([10, 10])
        self.vision = 5

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Dict({
            "relative_position": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "local_grid": spaces.Box(low=0.0, high=1.0, shape=(self.vision, self.vision), dtype=np.float32)
        })

        self.reset()

    def reset(self):
        self.position = np.array([0, 0])
        self.target = np.array([self.grid_size[0]-1, self.grid_size[1]-1])
        
        relative_position = (self.target - self.position) / self.grid_size
        local_grid = self.get_local_grid()

        return np.concatenate([relative_position, local_grid])
    
    def get_local_grid(self):

        return