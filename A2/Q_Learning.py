import numpy as np 
import math
from grid_world import WindyGridWorld as env

class Q_Learning():
    def __init__(self, episodes=10, lr=0.9, discount=0.9, epsilon=0.1, states=70, actions=4):
        self.episodes       = episodes
        self.learning_rate  = lr
        self.discount       = discount
        self.epsilon        = epsilon
        self.q_table        = np.zeros((states,actions))
        self.actions        = np.array([0,1,2,3]) #UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3
        self.env            = env()


    def eps_greedy_policy(self,state):
        """
            Takes in the size of the agent's action space (num_possible) and current state.
            Returns a policy pi where pi(argmax,state) = 1-epsilon 
            and pi(otheAction,state) = epsilon/|num_possible|
        """
        num_possible = len(self.actions)
        policy = np.ones(num_possible) * (self.epsilon / num_possible)
        q_index = self.__state_to_q_ind(state)
        best_action = np.argmax(self.q_table[q_index])
        policy[best_action] += 1.0 - self.epsilon

        return policy


    def __state_to_q_ind(self,state):
        """
            Takes in the state (row,col) and maps it an index in the q-table
        """
        n = self.q_table.shape[1] # num possible actions

        return (state[0]*n)+state[1]


    def __back_from_q_ind(self,state):
        """
            This one's for hugh and leonard...
        """
        n = self.q_table.shape[1] # num possible actions
        return math.floor(state[0]/n),state[0]%n

    def Q_learning(self):
        results = np.zeros(self.episodes) # Tracking results

        for episode in range(self.episodes):
            S = self.env.reset() # init S
            done = False
            print("Starting epsiode {}".format(episode+1))
            episode_value = 0
            while not done:
                policy = self.eps_greedy_policy(S)
                # Select epsilon greedy action and act
                action = np.random.choice(
                    np.arange(len(policy)),
                    p=self.eps_greedy_policy(S)
                )
                S_, done, R = self.env.act(action)
                episode_value = episode_value*self.discount + R
                q_index = self.__state_to_q_ind(S)
                # Update Q table
                
                greedy_next = np.argmax(self.q_table[self.__state_to_q_ind(S_)])
                old_val = self.q_table[q_index][action]
                old_val += self.learning_rate*(self.discount*greedy_next-old_val)
                S = S_
                
            results[episode] = episode_value
 
        # return results


if __name__ == "__main__":
    test = Q_Learning(10,0.9,0.9,0.1)
    print(test.Q_learning())