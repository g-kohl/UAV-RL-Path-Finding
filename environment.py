import gym
from gym import spaces
import numpy as np

class Environment(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 10
    }

    def __init__(self):
        super(Environment, self).__init__()

        self.grid_size = (10, 10)
        self.vision = 5

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Dict({
            "relative_position": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "local_grid": spaces.Box(low=0.0, high=1.0, shape=(self.vision, self.vision), dtype=np.float32)
        })

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size[0] + self.vision - 1, self.grid_size[1] + self.vision - 1))
        self.margin = self.vision // 2

        self.position = self.get_position()
        self.target = self.get_target()
        
        self.reset_grid()
        self.place_obstacles()

        return self.get_observation()
    
    def get_position(self):
        return np.array([self.margin, self.margin])
    
    def get_target(self):
        return np.array([self.margin + self.grid_size[0] - 1, self.margin + self.grid_size[1] - 1])
    
    def reset_grid(self):
        for i in range(self.margin):
            self.grid[i, :] = 1
            self.grid[self.grid_size[0]-1-i, :] = 1

            self.grid[:, i] = 1
            self.grid[:, self.grid_size[1]-1-i] = 1

    def place_obstacles(self):
        return #future implementation
    
    def get_observation(self):
        relative_position = (self.target - self.position) / np.array(self.grid_size)
        local_grid = self.get_local_grid()

        return {
            "relative_position": relative_position.astype(np.float32),
            "local_grid": local_grid.astype(np.float32)
        }
    
    def get_local_grid(self):
        return self.grid[self.position[0]-self.margin : self.position[0]+self.margin + 1,
                         self.position[1]-self.margin : self.position[1]+self.margin + 1]
    
    def step(self, action):
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        move = np.array(moves[action])
        next_position = self.position + move

        self.update_obstacles()

        if self.grid[next_position[0]][next_position[1]] == 1:
            reward = -100
            done = True
        elif np.array_equal(next_position, self.target):
            reward = 100
            done = True
        else:
            reward = -0.1
            done = False
            self.position = next_position

        return self.get_observation(), reward, done, {}
    
    def update_obstacles(self):
        return #future implementation
    
    def render(self, mode='human'):
        grid_display = self.grid.copy()
        x, y = self.position
        tx, ty = self.target

        grid_display[x, y] = 2
        grid_display[tx, ty] = 3

        if mode == 'human':
            print(grid_display)
        elif mode == 'ansi':
            return str(grid_display)
        
    def close(self):
        pass