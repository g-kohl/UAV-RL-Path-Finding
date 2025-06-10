import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class Environment(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 10
    }

    def __init__(self, render_mode=None):
        super(Environment, self).__init__()

        self.grid_size = (10, 10)
        self.vision = 5

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + self.vision * self.vision,), dtype=np.float32)

        self.reset()

        self.render_mode = render_mode
        self.cell_size = 50
        self.panel_height = 50

        self.window_size = (self.grid_size[0] * self.cell_size, self.grid_size[1] * self.cell_size + self.panel_height)

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
            pygame.display.set_caption("UAV Path Finding")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Ubuntu", 20)

    def reset(self, seed=None):
        self.grid = np.zeros((self.grid_size[0] + self.vision - 1, self.grid_size[1] + self.vision - 1))
        self.margin = self.vision // 2

        self.grid_coordinates = {
            "first" : np.array([self.margin, self.margin]),
            "last"  : np.array([self.margin + self.grid_size[0] - 1, self.margin + self.grid_size[1] - 1])
        }

        self.position = self.get_position(seed)
        self.target = self.get_target()
        
        self.reset_grid()
        self.place_obstacles()

        return self.get_observation(), {}
    
    def get_position(self, seed=None):
        return self.grid_coordinates["first"]
    
    def get_target(self):
        return self.grid_coordinates["last"]
    
    def reset_grid(self):
        for i in range(self.margin):
            self.grid[i, :] = 1
            self.grid[len(self.grid[0])-1-i, :] = 1

            self.grid[:, i] = 1
            self.grid[:, len(self.grid[1])-1-i] = 1

    def place_obstacles(self):
        pass #future implementation
    
    def get_observation(self):
        self.relative_position = (self.target - self.position) / np.array(self.grid_size)
        local_grid = self.get_local_grid().flatten()

        return np.concatenate([self.relative_position.astype(np.float32), local_grid.astype(np.float32)])
    
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
            terminated = True
        elif np.array_equal(next_position, self.target):
            reward = 100
            terminated = True
        else:
            reward = -0.1
            terminated = False

        self.position = next_position
        truncated = False

        return self.get_observation(), reward, terminated, truncated, {}
    
    def update_obstacles(self):
        pass #future implementation
    
    def render(self):
        if self.render_mode != 'human':
            return
        
        self.screen.fill((200, 200, 200))

        for i in range(self.vision):
            for j in range(self.vision):
                coordinates = (self.position[0] - self.margin + j, self.position[1] - self.margin + i)

                if not self.out_of_bounds(coordinates):
                    view_rect = pygame.Rect(
                        (coordinates[1] - self.margin) * self.cell_size,
                        (coordinates[0] - self.margin) * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )

                    pygame.draw.rect(self.screen, (255, 255, 255), view_rect)

        for i in range(self.grid_size[1]):
            for j in range(self.grid_size[0]):
                grid_rect = pygame.Rect(
                    i * self.cell_size,
                    j * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                pygame.draw.rect(self.screen, (150, 150, 150), grid_rect, 1)

        target_rect = pygame.Rect(
            (self.target[1] - self.margin) * self.cell_size,
            (self.target[0] - self.margin) * self.cell_size,
            self.cell_size,
            self.cell_size
        )

        pygame.draw.rect(self.screen, (255, 0, 0), target_rect)

        drone_rect = pygame.Rect(
            (self.position[1] - self.margin) * self.cell_size,
            (self.position[0] - self.margin) * self.cell_size,
            self.cell_size,
            self.cell_size
        )

        pygame.draw.rect(self.screen, (0, 255, 0), drone_rect)

        text_surface = self.font.render(str(self.relative_position), True, (0, 0, 0))
        self.screen.blit(text_surface, (15, self.grid_size[1] * self.cell_size + 15))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def out_of_bounds(self, coordinates):
        if (coordinates[0] < self.grid_coordinates["first"][0] or
            coordinates[1] < self.grid_coordinates["first"][1] or
            coordinates[0] > self.grid_coordinates["last"][0] or
            coordinates[1] > self.grid_coordinates["last"][0]):
            return True
        
        return False
        
    def close(self):
        if hasattr(self, "screen"):
            pygame.quit()
