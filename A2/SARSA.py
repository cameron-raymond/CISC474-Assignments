from Q_Learning import Q_Learning
import numpy as np
import matplotlib.pyplot as plt

class SARSA(Q_Learning):
    def __init__(self, shape=(7, 10), episodes=100, lr=0.9, discount=0.9, epsilon=0.1, king=False):
        super().__init__(shape=shape, episodes=episodes, lr=lr, discount=discount, epsilon=epsilon, king=king)
        self.title = "SARSA" if not king else "SARSA with Kings Move and Stochastic Wind"
    
    def train(self):
        """
            Apply SARSA to agent for a series of episodes
        """
        episode_steps = np.zeros(self.episodes)  # Tracking results
        for episode in range(self.episodes):
            state = self.env.reset()  # init S
            done = False
            steps = 0
            while not done and steps < 10000:
                # Get action, reward, and next state
                action = self.get_action(state)
                next_state, done, reward = self.env.act(action)
                q_index = self.state_to_q_ind(state)

                # Update state action table according to policy 
                next_action = self.get_action(next_state)
                next_state, _, _ = self.env.act(next_action)
                next_state_ind = self.state_to_q_ind(next_state)
                next_state_val = self.q_table[next_state_ind][next_action]
                old_val = self.q_table[q_index][action]
                update = self.learning_rate*(reward + self.discount*next_state_val-old_val)
                self.q_table[q_index][action] += update

                # Move to next time step
                state = next_state
                steps += 1

            # Print episode status
            if self.episodes < 10 or episode == 0 or (episode+1) % (self.episodes//10) == 0:
                print("    Episode {}\t- Total Steps: {}".format(episode+1,steps))

            episode_steps[episode] = steps
        return episode_steps
