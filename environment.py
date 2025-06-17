import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class Environment(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 10
    }

    def __init__(self, grid_size=(10, 10), static_obstacles=False, mobile_obstacles=False, seed=None, render_mode=None):
        super(Environment, self).__init__()

        self.grid_size = grid_size # (height, width) or (rows, columns)
        self.vision = 5

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + self.vision * self.vision,), dtype=np.float32)

        self.static_obstacles = static_obstacles
        self.mobile_obstacles = mobile_obstacles
        self.reset(seed)

        self.render_mode = render_mode
        self.cell_size = 50
        panel_height = 50

        window_width = self.grid_size[1] * self.cell_size
        window_height = self.grid_size[0] * self.cell_size + panel_height

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("UAV Path Finding")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Ubuntu", 20)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_steps = 0

        self.margin = self.vision // 2
        self.grid = self.reset_grid()

        self.grid_coordinates = {
            "first" : np.array([self.margin, self.margin]),
            "last"  : np.array([self.margin + self.grid_size[0] - 1, self.margin + self.grid_size[1] - 1])
        }

        self.place_obstacles()
        self.reset_position_and_target()

        return self.get_observation(), {}
    
    def reset_grid(self):
        grid = np.zeros((self.grid_size[0] + self.vision - 1, self.grid_size[1] + self.vision - 1))

        for i in range(self.margin):
            grid[i, :] = 1
            grid[grid.shape[0]-1-i, :] = 1

            grid[:, i] = 1
            grid[:, grid.shape[1]-1-i] = 1

        return grid
    
    def place_obstacles(self):
        if self.static_obstacles:
            middle_row = (self.grid_size[0] // 2) + self.margin
            wall = (4 + self.margin, self.grid_size[1] - 4 + self.margin)

            self.grid[middle_row, wall[0]:wall[1]] = 1

        if self.mobile_obstacles:
            self.mobile_obstacles_number = 2
            self.mobile_obstacles_positions = []

            for _ in range(self.mobile_obstacles_number):
                while True:
                    position = self.get_random_coordinates()

                    if not self.detect_obstacle(position):
                        break

                self.mobile_obstacles_positions.append(position)
                self.grid[self.margin + position[0]][self.margin + position[1]] = 1

    
    def reset_position_and_target(self):
        while True:
            self.position = self.get_random_coordinates()
            self.target = self.get_random_coordinates()

            if not np.array_equal(self.position, self.target) and not self.detect_obstacle(self.position) and not self.detect_obstacle(self.target):
                break

    def get_random_coordinates(self):
        return np.array([
            self.np_random.integers(self.grid_coordinates["first"][0], self.grid_coordinates["last"][0]),
            self.np_random.integers(self.grid_coordinates["first"][1], self.grid_coordinates["last"][1])
        ])
    
    def detect_obstacle(self, position):
        if self.grid[position[0] + self.margin][position[1] + self.margin] == 1:
            return True
        
        return False
    
    def get_observation(self):
        self.relative_position = (self.target - self.position) / np.array(self.grid_size)
        local_grid = self.get_local_grid().flatten()

        return np.concatenate([self.relative_position.astype(np.float32), local_grid.astype(np.float32)])
    
    def get_local_grid(self):
        return self.grid[self.position[0]-self.margin : self.position[0]+self.margin + 1,
                         self.position[1]-self.margin : self.position[1]+self.margin + 1]
    
    def step(self, action):
        self.current_steps += 1

        if self.current_steps > 100:
            reward = -100
            truncated = True
        else:
            truncated = False

        self.moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        move = np.array(self.moves[action])
        next_position = self.position + move

        self.update_obstacles()

        if self.detect_obstacle(next_position):
            reward = -100
            terminated = True
        elif np.array_equal(next_position, self.target):
            reward = 100
            terminated = True
            self.position = next_position
        else:
            reward = -0.1
            terminated = False
            self.position = next_position

        return self.get_observation(), reward, terminated, truncated, {}
    
    def update_obstacles(self):
        if not self.mobile_obstacles:
            return

        for obstacle in range(self.mobile_obstacles_number):
            while True:
                action = random.randrange(0, 8)
                move = np.array(self.moves[action])

                if not self.detect_obstacle(self.mobile_obstacles_positions[obstacle] + move):
                    self.grid[self.margin + self.mobile_obstacles_positions[obstacle][0]][self.margin + self.mobile_obstacles_positions[obstacle][1]] = 0
                    self.mobile_obstacles_positions[obstacle] += move
                    self.grid[self.margin + self.mobile_obstacles_positions[obstacle][0]][self.margin + self.mobile_obstacles_positions[obstacle][1]] = 1
                    break
    
    def render(self):
        if self.render_mode != 'human':
            return
        
        self.screen.fill((200, 200, 200))

        # draw uav vision
        for i in range(self.vision):
            for j in range(self.vision):
                coordinates = (self.position[0] - self.margin + i, self.position[1] - self.margin + j)

                if not self.out_of_bounds(coordinates):
                    view_rect = pygame.Rect(
                        (coordinates[1] - self.margin) * self.cell_size,
                        (coordinates[0] - self.margin) * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )

                    pygame.draw.rect(self.screen, (255, 255, 255), view_rect)

        # draw grid
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                grid_rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                pygame.draw.rect(self.screen, (150, 150, 150), grid_rect, 1)

        # draw obstacles
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.detect_obstacle((i, j)):
                    obstacle_rect = pygame.Rect(
                        j * self.cell_size,
                        i * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )

                    pygame.draw.rect(self.screen, (0, 0, 0), obstacle_rect)

        # draw target
        target_rect = pygame.Rect(
            (self.target[1] - self.margin) * self.cell_size,
            (self.target[0] - self.margin) * self.cell_size,
            self.cell_size,
            self.cell_size
        )

        pygame.draw.rect(self.screen, (255, 0, 0), target_rect)

        # draw uav
        drone_rect = pygame.Rect(
            (self.position[1] - self.margin) * self.cell_size,
            (self.position[0] - self.margin) * self.cell_size,
            self.cell_size,
            self.cell_size
        )

        pygame.draw.rect(self.screen, (0, 255, 0), drone_rect)

        # draw text background
        text_background_rect = pygame.Rect(
            0,
            self.grid_size[0] * self.cell_size,
            self.grid_size[1] * self.cell_size,
            self.cell_size
        )

        pygame.draw.rect(self.screen, (20, 20, 20), text_background_rect)

        # draw text
        text = self.font.render(f"Relative position: {np.round(self.relative_position, 2)}", True, (255, 255, 255))
        self.screen.blit(text, (15, self.grid_size[0] * self.cell_size + 15))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def out_of_bounds(self, coordinates):
        if (coordinates[0] < self.grid_coordinates["first"][0] or
            coordinates[1] < self.grid_coordinates["first"][1] or
            coordinates[0] > self.grid_coordinates["last"][0] or
            coordinates[1] > self.grid_coordinates["last"][1]):
            return True
        
        return False
        
    def close(self):
        if hasattr(self, "screen"):
            pygame.quit()
