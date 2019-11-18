import numpy as np 
import math
from grid_world import WindyGridWorld as env

class Q_Learning():
    def __init__(self, shape=(7,10), episodes=100, lr=0.9, discount=0.9, epsilon=0.1, actions=4):
        self.episodes       = episodes
        self.shape          = shape
        self.learning_rate  = lr
        self.discount       = discount
        self.epsilon        = epsilon
        self.states         = self.__state_to_q_ind(shape)
        self.q_table        = np.zeros((self.states,actions))
        self.actions        = np.array([0,1,2,3]) #UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3
        self.env            = env(shape)
        self.terminal       = self.env.terminal # note take out so our code doesn't look jank


    def eps_greedy_policy(self,state):
        """
            Takes in the size of the agent's action space (num_possible) and current state.
            Returns a policy pi where pi(argmax,state) = 1-epsilon + epsilon/|num_possible|
            and pi(otheAction,state) = epsilon/|num_possible|
        """
        q_index = self.__state_to_q_ind(state)
        num_possible = len(self.actions)
        
        policy = np.ones(num_possible) * (self.epsilon / num_possible)
        best_action = np.argmax(self.q_table[q_index])
        policy[best_action] += 1.0 - self.epsilon
        if len(set(self.q_table[q_index])) == 1:
            print("all choices equal => random walk")
            # If all the elements are equal, random walk
            policy = np.ones(num_possible) / num_possible
        print(policy)


        return policy


    def __state_to_q_ind(self,state):
        """
            Takes in the state (row,col) and maps it an index in the q-table
        """
        n = self.shape[1] # num possible actions
        return int((state[0]*n)+state[1])
    
    def print_state(self,state):      
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                q_table_ind = self.__state_to_q_ind((row,col))
                q_table_max = max(self.q_table[q_table_ind])
                q_table_min = min(self.q_table[q_table_ind])
                if (row,col) == state:
                    print("X",end='\t|')
                elif (row,col) == self.env.terminal:
                    print("T",end='\t|')
                else:
                    print("o",end='\t|')
            print("\n")
        print("---")

    def print_q_table(self):
        print(*["S","U","R","D","L"],sep="\t|")
        for ind,state in enumerate(self.q_table):
            to_tuple = self.back_from_q_ind(ind)
            print(to_tuple,end='\t|')
            for sorry in state:
                print("{:0.2f}\t|".format(sorry),end="")
            print('\n')

    def back_from_q_ind(self,state):
        """
            This one's for hugh and leonard...
        """
        n = self.shape[1] # num possible actions
        return math.floor(state/n),state%n

    def Q_learning(self):
        results = np.zeros(self.episodes) # Tracking results

        for episode in range(self.episodes):
            S = self.env.reset() # init S
            done = False
            episode_value = 0
            while not done:
                # self.print_state(S)
                # self.print_q_table()
                policy = self.eps_greedy_policy(S)
                # Select epsilon greedy action and act
                action = np.random.choice(
                    np.arange(len(policy)),
                    p=policy
                )
                S_, done, R = self.env.act(action)
                assert(R != 0 or S_ == self.env.terminal)
                q_index = self.__state_to_q_ind(S)
                # Update Q table
                S_q_ind = self.__state_to_q_ind(S_)
                greedy_next = np.max(self.q_table[S_q_ind])
                old_val = self.q_table[q_index][action]
                # print("Q(S,A) = {:0.2f} + {}[{}+{}*{:0.2f}-{:0.2f}]".format(old_val,self.learning_rate,R,self.discount,greedy_next,old_val))
                update = self.learning_rate*(R + self.discount*greedy_next-old_val)
                self.q_table[q_index][action] += update
                S = S_

                episode_value += episode_value*self.discount + R
            
            if self.episodes < 10 or episode == 0 or (episode+1) % (self.episodes//10) == 0:
                print("value of terminal state: {}".format(self.q_table[self.terminal]))
                print("Gt for episode {}: {}".format(episode+1,episode_value))
                
            results[episode] = episode_value
            self.print_q_table()
 
        return results


if __name__ == "__main__":
    test = Q_Learning(episodes=10,lr=0.1,discount=0.2,epsilon=0)
    print(test.Q_learning())