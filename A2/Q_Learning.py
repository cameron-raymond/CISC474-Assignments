import numpy as np 
import math
from grid_world import WindyGridWorld as env
import matplotlib.pyplot as plt

class Q_Learning(object):
    def __init__(self, shape=(7,10), episodes=100, lr=0.9, discount=0.9, epsilon=0.1, king=False,):
        """ 
        Possible moves: 
            UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, UP and LEFT = 4, UP and RIGHT = 5, DOWN and RIGHT = 6, DOWN and LEFT = 7 
        """
        self.title          = "Q Learning" if not king else "Q Learning with Kings Move and Stochastic Wind"
        self.episodes       = episodes
        self.shape          = shape
        self.learning_rate  = lr
        self.discount       = discount
        self.epsilon        = epsilon
        self.states         = self.state_to_q_ind(shape)
        self.possible_moves = 8 if king else 4 
        self.q_table        = np.zeros((self.states,self.possible_moves))
        self.actions        = np.array(range(self.possible_moves))  
        self.env            = env(shape=shape, stochastic_wind=king)

    def eps_greedy_policy(self,state):
        """
            Takes in the size of the agent's action space (num_possible) and current state.
            Returns a policy pi where pi(argmax,state) = 1-epsilon + epsilon/|num_possible|
            and pi(otheAction,state) = epsilon/|num_possible|
        """
        q_index = self.state_to_q_ind(state)
        num_possible = len(self.actions)
        policy = np.ones(num_possible) * (self.epsilon / num_possible)
        best_action = np.argmax(self.q_table[q_index])
        policy[best_action] += 1.0 - self.epsilon
        if len(set(self.q_table[q_index])) == 1:
            # If all the elements are equal, random walk
            policy = np.ones(num_possible) / num_possible
        return policy

    def get_action(self,state):
        policy = self.eps_greedy_policy(state)
        action = np.random.choice(np.arange(len(policy)),p=policy)
        return action

    def state_to_q_ind(self,state):
        """
            Takes in the state (row,col) and maps it an index in the q-table
        """
        n = self.shape[1] # each row in the grid has n possible columns that it could be in
        return int((state[0]*n)+state[1])

    def train(self):
        episode_steps = np.zeros(self.episodes)# Tracking results
        for episode in range(self.episodes):
            state = self.env.reset() # init S
            done = False
            steps = 0
            while not done and steps < 10000:
                action = self.get_action(state)
                next_state, done, reward = self.env.act(action)
                q_index = self.state_to_q_ind(state)
                # Update Q table
                S_q_ind = self.state_to_q_ind(next_state)
                greedy_next = np.max(self.q_table[S_q_ind])
                old_val = self.q_table[q_index][action]
                update = self.learning_rate*(reward + self.discount*greedy_next-old_val)
                self.q_table[q_index][action] += update
                state = next_state
                steps += 1
            if self.episodes < 10 or episode == 0 or (episode+1) % (self.episodes//10) == 0:
                print("    Episode {}\t- Total Steps: {}".format(episode+1,steps))
            episode_steps[episode] = steps 
        return episode_steps

    def __str__(self):
        """ 
            A string representation of the state action table
        """
        to_tuple = lambda q_index : (math.floor(q_index/self.shape[1]),q_index%self.shape[1]) # converts the Q table index back to the original state tuple
        q_table = "\t<--- {} Q Table --->\n".format(self.title)
        q_table += "S\t| U\t| R\t| D\t| L\t|\n"
        for ind,state in enumerate(self.q_table):
            s_tuple = to_tuple(ind)
            state_str  = "({t[0]},{t[1]})\t| {s[0]:0.2f}\t| {s[1]:0.2f}\t| {s[2]:0.2f}\t| {s[3]:0.2f}\t|\n".format(t=s_tuple,s=state)
            q_table += state_str
        return q_table

    def plot(self,episode_steps):
        plt.plot(episode_steps)
        plt.title('{}: Episodes vs steps'.format(self.title))
        plt.xlabel('Episodes')
        plt.ylabel('steps')
        plt.show()
    
