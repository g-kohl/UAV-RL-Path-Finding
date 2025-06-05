# drone_env.py
import gym
from gym import spaces
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self, grid_size=(10, 10, 3), vision=3):
        super().__init__()
        self.grid_size = grid_size
        self.vision = vision  # tamanho da janela local de observação (ex: 3x3x3)
        self.action_space = spaces.Discrete(6)  # frente/trás/esq/dir/cima/baixo

        # Estado = [x_rel, y_rel, z_rel] + grade local achatada
        obs_len = 3 + (vision ** 3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.pos = np.array([0, 0, 0])
        self.goal = np.array([self.grid_size[0]-1, self.grid_size[1]-1, self.grid_size[2]-1])
        self.grid = np.zeros(self.grid_size)
        self._place_obstacles()
        return self._get_obs()

    def _place_obstacles(self):
        # Exemplo: adiciona obstáculos aleatórios
        for _ in range(15):
            obs = tuple(np.random.randint(0, s) for s in self.grid_size)
            if not np.array_equal(obs, self.goal):
                self.grid[obs] = 1

    def _get_obs(self):
        rel_pos = (self.goal - self.pos) / np.array(self.grid_size)
        local_grid = self._get_local_grid()
        return np.concatenate([rel_pos, local_grid.flatten()])

    def _get_local_grid(self):
        x, y, z = self.pos
        v = self.vision // 2
        pad_grid = np.pad(self.grid, v, constant_values=1)  # borda como parede
        x, y, z = x+v, y+v, z+v
        return pad_grid[x-v:x+v+1, y-v:y+v+1, z-v:z+v+1]

    def step(self, action):
        moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        move = np.array(moves[action])
        next_pos = self.pos + move

        done = False
        reward = -0.1  # penalidade padrão

        if not self._in_bounds(next_pos):
            reward = -100
            done = True
        elif self.grid[tuple(next_pos)] == 1:
            reward = -100
            done = True
        elif np.array_equal(next_pos, self.goal):
            reward = 100
            done = True
        else:
            self.pos = next_pos

        return self._get_obs(), reward, done, {}

    def _in_bounds(self, pos):
        return all(0 <= pos[i] < self.grid_size[i] for i in range(3))

    def render(self, mode='human'):
        print(f"Posição: {self.pos}, Destino: {self.goal}")
