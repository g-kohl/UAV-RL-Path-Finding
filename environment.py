import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

UAV_VISION = 5
INITIAL_STEPS = 0 # based on the previous training of the current model
MAX_WINDOW_SIZE = 600
PANEL_HEIGHT = 50
MAPS = 8

class Environment(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 10
    }

    def __init__(self, grid_size=(10, 10), static_obstacles=False, mobile_obstacles=False, training=True, seed=None, render_mode=None):
        super(Environment, self).__init__()

        self.grid_size = grid_size # (height, width) or (rows, columns)

        self.action_space = spaces.Discrete(8)
        self.moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + UAV_VISION * UAV_VISION,), dtype=np.float32)

        self.static_obstacles = static_obstacles
        self.mobile_obstacles = mobile_obstacles
        self.training = training

        self.total_steps = INITIAL_STEPS
        self.reset(seed)

        self.render_mode = render_mode
        self.cell_size = MAX_WINDOW_SIZE / max(grid_size[0], grid_size[1])
        window_height = self.grid_size[0] * self.cell_size + PANEL_HEIGHT
        window_width = self.grid_size[1] * self.cell_size

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("UAV Path Finding")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Ubuntu", 20)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_steps = 0

        self.margin = UAV_VISION // 2
        self.grid = self.reset_grid()

        self.grid_coordinates = {
            "first" : np.array([self.margin, self.margin]),
            "last"  : np.array([self.margin + self.grid_size[0] - 1, self.margin + self.grid_size[1] - 1])
        }

        self.danger_coordinates = []

        if self.static_obstacles:
            self.place_static_obstacles()

        if self.mobile_obstacles:
            self.place_mobile_obstacles()

        self.reset_position_and_target()

        return self.get_observation(), {}
    

    def reset_grid(self):
        grid = np.zeros((self.grid_size[0] + UAV_VISION - 1, self.grid_size[1] + UAV_VISION - 1))

        for i in range(self.margin): # place walls on the borders of the map
            grid[i, :] = 1
            grid[grid.shape[0]-1-i, :] = 1

            grid[:, i] = 1
            grid[:, grid.shape[1]-1-i] = 1

        return grid
    

    def place_static_obstacles(self):
        if self.grid_size != (15, 15): # all maps are 15x15
            print("NO STATIC OBSTACLES: grid and obstacles map have different sizes")
            return

        difficulty = self.select_difficulty()
        map_id = self.np_random.integers(1, MAPS + 1)

        if self.training:
            path = f"maps/training/{difficulty}/map_"
        else:
            path = f"maps/testing/map_"

        with open(f"{path}{map_id}.txt") as map: # load obstacles based on the map file
            self.free_space = 0

            for i, line in enumerate(map):
                line = "".join([char for char in line if char != "\n" and char != " "])

                for j, char in enumerate(line):
                    grid_coodinates = (i + self.grid_coordinates["first"][0], j + self.grid_coordinates["first"][1])

                    if char == '2': # danger coordinates
                        self.danger_coordinates.append((grid_coodinates[0], grid_coodinates[1]))

                    char = 1 if char == '1' else 0 # other numbers in the map are converted to zero
                    self.grid[grid_coodinates[0]][grid_coodinates[1]] = char
                    
                    if char != 1:
                        self.free_space += 1


    def select_difficulty(self):
        if self.total_steps < 1_000_000:
            return "easy"
        if self.total_steps < 3_000_000:
            return "medium"
        
        return "hard"


    def place_mobile_obstacles(self):
        self.mobile_obstacles_number = self.free_space // 50 # good fraction of mobile obstacles
        self.mobile_obstacles_positions = []

        for _ in range(self.mobile_obstacles_number):
            while True:
                position = self.get_random_coordinates()

                if not self.detect_obstacle(position):
                    break

            self.mobile_obstacles_positions.append(position)
            self.grid[position[0]][position[1]] = 1


    def get_random_coordinates(self):
        return np.array([
            self.np_random.integers(self.grid_coordinates["first"][0], self.grid_coordinates["last"][0] + 1),
            self.np_random.integers(self.grid_coordinates["first"][1], self.grid_coordinates["last"][1] + 1)
        ])
    
    
    def get_random_danger_coordinates(self):
        coordinates = self.danger_coordinates[self.np_random.integers(0, len(self.danger_coordinates))]

        return np.array(coordinates)


    def detect_obstacle(self, position):
        if self.grid[position[0]][position[1]] == 1:
            return True
        
        return False

    
    def reset_position_and_target(self):
        while True:
            self.position = self.get_random_coordinates()

            if self.np_random.random() < 0.75 and len(self.danger_coordinates) > 0: # 75% of the episodes, the target will be close to a wall
                self.target = self.get_random_danger_coordinates()
            else:
                self.target = self.get_random_coordinates()

            if not np.array_equal(self.position, self.target) and not self.detect_obstacle(self.position) and not self.detect_obstacle(self.target):
                break

    
    def get_observation(self):
        self.relative_position = (self.target - self.position) / np.array(self.grid_size)
        local_grid = self.get_local_grid().flatten()

        return np.concatenate([self.relative_position.astype(np.float32), local_grid.astype(np.float32)])
    

    def get_local_grid(self):
        return self.grid[self.position[0]-self.margin : self.position[0]+self.margin + 1,
                         self.position[1]-self.margin : self.position[1]+self.margin + 1]
    

    def step(self, action):
        self.total_steps += 1
        self.current_steps += 1

        if self.current_steps > self.grid_size[0] * self.grid_size[1]: # UAV took many steps (more steps than coordinates in the grid)
            reward = -100
            terminated = False
            truncated = True
        else:
            truncated = False

        move = np.array(self.moves[action])
        next_position = self.position + move

        self.update_obstacles()

        if not truncated:
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

        reward = np.clip(reward / 100, -1.0, 1.0)

        return self.get_observation(), reward, terminated, truncated, {}
    

    def update_obstacles(self):
        if not self.mobile_obstacles:
            return

        for i, obstacle_position in enumerate(self.mobile_obstacles_positions):
            directions = list(range(8))
            self.np_random.shuffle(directions)

            for action in directions:
                move = np.array(self.moves[action])

                if not self.detect_obstacle(obstacle_position + move):
                    self.grid[self.mobile_obstacles_positions[i][0]][self.mobile_obstacles_positions[i][1]] = 0
                    self.mobile_obstacles_positions[i] += move
                    self.grid[self.mobile_obstacles_positions[i][0]][self.mobile_obstacles_positions[i][1]] = 1
                    break
    

    def render(self):
        if self.render_mode != 'human':
            return
        
        self.screen.fill((200, 200, 200))

        # draw uav vision
        for i in range(UAV_VISION):
            for j in range(UAV_VISION):
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

        # draw target
        target_rect = pygame.Rect(
            (self.target[1] - self.margin) * self.cell_size,
            (self.target[0] - self.margin) * self.cell_size,
            self.cell_size,
            self.cell_size
        )

        pygame.draw.rect(self.screen, (255, 0, 0), target_rect)

        # draw obstacles
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.detect_obstacle((i + self.grid_coordinates["first"][0], j + self.grid_coordinates["first"][1])):
                    obstacle_rect = pygame.Rect(
                        j * self.cell_size,
                        i * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )

                    pygame.draw.rect(self.screen, (0, 0, 0), obstacle_rect)

        # draw uav
        UAV_rect = pygame.Rect(
            (self.position[1] - self.margin) * self.cell_size,
            (self.position[0] - self.margin) * self.cell_size,
            self.cell_size,
            self.cell_size
        )

        pygame.draw.rect(self.screen, (0, 255, 0), UAV_rect)

        # draw text background
        text_background_rect = pygame.Rect(
            0,
            self.grid_size[0] * self.cell_size,
            self.grid_size[1] * self.cell_size,
            PANEL_HEIGHT
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
