

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridWorld(object):
    """Create an environment of a Grid World
    R = dict {(s):reward} - reward matrix, reward which obtained on state s
    """

    def __init__(self, shape=(7, 10),  start=(3, 0), terminal=(3, 7),
                    wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]):
        self.shape = shape
        self.R = {}
        self.terminal = terminal
        self.state = start
        self.wind = [0, 0, 0, 0,0,0,0,0,0, 0]

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.R[(i, j)] = 0 if (i, j)==self.terminal else -1.0
    
    def reset(self):
        self.state = (3, 0)
        return self.state

    def act(self, action):
        if action==UP:
            self.act_up()
        if action==RIGHT:
            self.act_right()
        if action==LEFT:
            self.act_left()
        if action==DOWN:
            self.act_down()
        self.act_up(step=self.wind[self.state[1]])
        return (self.state, self.state==self.terminal, self.R[self.state])

    def act_up(self, step=1):
        self.state = (max(self.state[0]-step, 0), self.state[1])

    def act_down(self, step=1):
        self.state = (min(self.state[0]+step, self.shape[0]-1), self.state[1])

    def act_right(self, step=1):
        self.state = (self.state[0], min(self.state[1]+step, self.shape[1]-1))

    def act_left(self, step=1):
        self.state = (self.state[0], max(self.state[1]-step, 0))
