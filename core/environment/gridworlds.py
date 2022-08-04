import random

import numpy as np

from core.utils.torch_utils import random_seed

class GridHardXY:
    def __init__(self, seed=np.random.randint(int(1e5))):
        self.rng = np.random.RandomState(seed)
        self.state_dim = (2,)
        self.action_dim = 4
        self.obstacles_map = self.get_obstacles_map()
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)] #right, left, down, up
        self.action_explaination = ["R", "L", "D", "U"]
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14
        self.goal_x, self.goal_y = 9, 9
        self.current_state = None

    def generate_state(self, coords):
        return np.array(coords)

    def info(self, key):
        return

    def reset(self):
        while True:
            rand_state = self.rng.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y):
                self.current_state = rand_state[0], rand_state[1]
                return self.generate_state(self.current_state)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy
        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)
        if not self.obstacles_map[nx][ny]:
            x, y = nx, ny

        self.current_state = x, y
        if x == self.goal_x and y == self.goal_y:
            return self.generate_state([x, y]), np.asarray(1.0), np.asarray(True), ""
        else:
            return self.generate_state([x, y]), np.asarray(0.0), np.asarray(False), ""

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(15)
                       for y in range(15) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        # goal_coords = [[9, 9], [0, 0], [14, 0], [7, 14]]
        # goal_coords = [[9, 9], [9, 12], [13, 11], [8, 6], [13, 2]] # 0 5 25 50 100
        # goal_coords = [[9, 9], [13, 11], [8, 6], [9, 1], [13, 2], [4, 13], [1, 3]] # 0 25 50 75 100 125 150
        goal_coords = [[9, 9], [3, 4], [1, 0], [0, 14], [3, 14], [7, 14], [10, 3], [14, 4]] #
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[2, 0:6] = 1.0
        _map[2, 8:] = 1.0
        _map[3, 5] = 1.0
        _map[4, 5] = 1.0
        _map[5, 2:7] = 1.0
        _map[5, 9:] = 1.0
        _map[8, 2] = 1.0
        _map[8, 5] = 1.0
        _map[8, 8:] = 1.0
        _map[9, 2] = 1.0
        _map[9, 5] = 1.0
        _map[9, 8] = 1.0
        _map[10, 2] = 1.0
        _map[10, 5] = 1.0
        _map[10, 8] = 1.0
        _map[11, 2:6] = 1.0
        _map[11, 8:12] = 1.0
        _map[12, 5] = 1.0
        _map[13, 5] = 1.0
        _map[14, 5] = 1.0

        return _map

    def get_useful(self, state=None):
        if state:
            return state
        else:
            return self.current_state
        
    def get_state_space(self):
        obs = self.get_obstacles_map()
        empty = np.argwhere(obs == 0)
        obs = np.argwhere(obs != 0)
        return empty, obs
    
    def get_goal_coord(self):
        return [self.goal_x, self.goal_y]

class FourRoom(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(FourRoom, self).__init__(seed)
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 10, 0, 10
        self.goal_x, self.goal_y = 1, 9
        self.obstacles_map = self.get_obstacles_map()

    def get_obstacles_map(self):
        _map = np.zeros([11, 11])
        _map[5, 0] = 1.0
        _map[5, 2:6] = 1.0
        _map[0:2, 5] = 1.0
        _map[3:9, 5] = 1.0
        _map[10, 5] = 1.0
        _map[6, 6:8] = 1.0
        _map[6, 9:11] = 1.0
        return _map

    # def reset(self):
    #     self.current_state = 10, 0
    #     return self.generate_state(self.current_state)

    def reset(self):
        while True:
            rand_state = self.rng.randint(low=0, high=11, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y):
                self.current_state = rand_state[0], rand_state[1]
                return self.generate_state(self.current_state)
