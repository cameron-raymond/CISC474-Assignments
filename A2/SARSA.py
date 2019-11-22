from Q_Learning import Q_Learning
import numpy as np
import matplotlib.pyplot as plt

class SARSA(Q_Learning):
    def __init__(self, shape=(7, 10), episodes=100, lr=0.9, discount=0.9, epsilon=0.1, king=False, _lambda=0):
        super().__init__(shape=shape, episodes=episodes, lr=lr, discount=discount, epsilon=epsilon, king=king, _lambda=_lambda)
        using_king = " with Kings Move and Stochastic Wind" if king else ""
        using_eligibility_trace = " using eligibility trace" if _lambda else ""

        self.title = "SARSA" + using_king + using_eligibility_trace
    
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

    def train_lambda(self):
        """
            Apply SARSA to agent for a series of episodes using eligibility traces
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

                # Udate E table
                next_action = self.get_action(next_state)
                next_state, _, _ = self.env.act(next_action)
                next_state_ind = self.state_to_q_ind(next_state)
                next_state_val = self.q_table[next_state_ind][next_action]
                old_val = self.q_table[q_index][action]
                target = reward + self.discount * next_state 
                error = target - old_val

                self.e_table += 1

                # Update Q table
                for s in range(len(self.q_table)):
                    for a in range(len(self.q_table[s])):
                        # Update Q table value based on eligibility trace
                        self.q_table[s][a] += self.learning_rate * error * self.e_table[s][a]
                        
                        # Decay eligibility trace if best action is taken
                        if next_state is np.max(self.q_table[self.state_to_q_ind(next_state)]):
                            self.e_table[s][a] = self.discount * self._lambda * self.e_table[s][a]
                        # Reset value if we've taken the random action
                        else:
                            self.e_table[s][a] = 0

                # Move to next time step
                state = next_state
                steps += 1

            self.e_table = np.zeros((self.states,self.possible_moves)) # re-init

            # Print episode status
            if self.episodes < 10 or episode == 0 or (episode+1) % (self.episodes//10) == 0:
                print("    Episode {}\t- Total Steps: {}".format(episode+1,steps))

            episode_steps[episode] = steps
        return episode_steps
