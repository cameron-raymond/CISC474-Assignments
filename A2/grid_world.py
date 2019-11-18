

import numpy as np
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
UP_LEFT = 4
UP_RIGHT = 5
DOWN_RIGHT = 6
DOWN_LEFT = 7

class WindyGridWorld(object):
    """Create an environment of a Grid World
    R = dict {(s):reward} - reward matrix, reward which obtained on state s
    """

    def __init__(self, shape=(7, 10),  start=(0, 0), terminal=(5,3),
                    wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0], stochastic_wind=False):
        self.shape = shape
        self.R = {}
        self.terminal = terminal
        self.state = start
        self.start_state = start
        self.wind = wind # np.zeros(shape[1]) #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.stochastic_wind = stochastic_wind

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.R[(i, j)] = 0 if (i, j)==self.terminal else -1.0
    
    def reset(self):
        self.state = self.start_state
        return self.state

    def act(self, action):
        if action == UP:
            self.act_up()
        if action == RIGHT:
            self.act_right()
        if action == LEFT:
            self.act_left()
        if action == DOWN:
            self.act_down()
        if action == UP_LEFT:
            self.act_up()
            self.act_left()
        if action == UP_RIGHT:
            self.act_up()
            self.act_right()
        if action == DOWN_RIGHT:
            self.act_down()
            self.act_right()
        if action == DOWN_LEFT:
            self.act_down()
            self.act_left()

        if self.stochastic_wind:
            this_wind = np.random.choice(np.array([
                self.wind[self.state[1]] - 1,
                self.wind[self.state[1]],
                self.wind[self.state[1]] + 1,
            ]))
        else:
            this_wind = self.wind[self.state[1]]
        self.act_up(this_wind)

        return (self.state, self.state==self.terminal, self.R[self.state])

    def act_up(self, step=1):
        self.state = (max(self.state[0]-step, 0), self.state[1])

    def act_down(self, step=1):
        self.state = (min(self.state[0]+step, self.shape[0]-1), self.state[1])

    def act_right(self, step=1):
        self.state = (self.state[0], min(self.state[1]+step, self.shape[1]-1))

    def act_left(self, step=1):
        self.state = (self.state[0], max(self.state[1]-step, 0))
