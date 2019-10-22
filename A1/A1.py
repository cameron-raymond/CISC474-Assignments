import numpy as np
from numpy.core.multiarray import ndarray

from visualizations import visualize_probabilities, opposing_probabilities


class PolicyIteration(object):
    def __init__(self, p1_reward, p2_reward, p1_policy, p2_policy, alpha, k, action_dict=None, second_algo_enable=False):
        super().__init__()
        self.p1_reward              = p1_reward
        self.p2_reward              = p2_reward
        self.actions                = [i for i in range(p1_reward.shape[0])]
        self.p1_policy              = p1_policy
        self.p2_policy              = p2_policy
        self.p1_policy_expectation  = self.p1_policy
        self.p2_policy_expectation  = self.p2_policy
        self.alpha                  = alpha
        self.k                      = k
        self.action_dict            = action_dict
        self.second_algo_enable     = second_algo_enable

    def softmax(self, x):
        """"
            Takes an n dimensional vector of real numbers, and normalizes it into a probability distribution 
            consisting of n probabilities proportional to the exponentials of the input numbers.
            https://en.wikipedia.org/wiki/Softmax_function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def train(self):
        action_dict = self.action_dict
        # Keep track of the probabilites over time to then plot
        p1_over_time = np.array([self.p1_policy])
        p2_over_time = np.array([self.p2_policy])
        for iteration in range(self.k):
            p1_action = np.random.choice(self.actions, p=self.p1_policy)
            p2_action = np.random.choice(self.actions, p=self.p2_policy)
            if self.k < 10 or iteration == 0 or (iteration+1) % (self.k//10) == 0:
                print("Iteration: {}".format(iteration+1))
                print("Player one policy {}".format(self.p1_policy))
                print("Player two policy {}".format(self.p2_policy))
                print("Player one {} and player two {}.".format(action_dict[p1_action], action_dict[p2_action]))
                print("Player one reward: {}. Player two reward: {}".format(self.p1_reward[p1_action, p2_action],
                                                                            self.p2_reward[p2_action, p1_action]))
            self.p1_policy, self.p1_policy_expectation = self._update_policy(self.p1_policy, self.p1_policy_expectation,
                                                                             self.p1_reward, p1_action, p2_action)
            self.p2_policy, self.p2_policy_expectation = self._update_policy(self.p2_policy, self.p2_policy_expectation,
                                                                             self.p2_reward, p2_action, p1_action)
            p1_over_time = np.vstack([p1_over_time, self.p1_policy])
            p2_over_time = np.vstack([p2_over_time, self.p2_policy])
        # Calculate value of game for player 1: P1*R1*P2^T
        p1_game_value = self.p1_policy.dot(self.p1_reward.dot(self.p2_policy.T))
        # Calculate value of game for player 2: P2*R2*P1^T
        p2_game_value = self.p2_policy.dot(self.p2_reward.dot(self.p1_policy.T))
        print("Game value for player 1: {}".format(p1_game_value))
        print("Game value for player 2: {}".format(p2_game_value))
        return p1_over_time, p2_over_time

    def _update_policy(self, policy, policy_expectation, reward_mat, user_act, opponent_act):
        # reward for an action also depends on what the other user did
        reward = reward_mat[user_act, opponent_act]
        action_probability = policy[user_act]
        # from equation 𝑝(𝑘 + 1) = 𝑝(𝑘) + 𝛼𝑟(𝑘)(1 − 𝑝(𝑘))
        new_action_probability = action_probability + (self.alpha*reward*(1-action_probability))
        assert (reward < 0 and (new_action_probability-action_probability) <= 0) or \
               (reward > 0 and (new_action_probability-action_probability) >= 0) or \
               (reward == 0 and new_action_probability == action_probability),\
            "ACTION: {}, OPP ACTION: {}, REWARD: {}, OLD PROB {:.4f}, NEW PROB: {:.4f} "\
                .format(self.action_dict[user_act], self.action_dict[opponent_act], reward, action_probability, new_action_probability)
        # update all other probabilities 𝑝(𝑘 + 1) = 𝑝(𝑘) − 𝛼𝑟(𝑘)*𝑝(𝑘) = 𝑝(𝑘) + −1*𝛼𝑟(𝑘)*𝑝(𝑘), 𝑓𝑜𝑟 𝑎𝑙𝑙 𝑜𝑡h𝑒𝑟 𝑎𝑐𝑡𝑖𝑜𝑛𝑠 𝑜 ≠ 𝑐
        # this is done by multiplying the policy vector by scalars alpha, the action reward and -1
        policy = policy-(self.alpha*reward*policy)
        # bring in the delta from the action we actually chose
        policy[user_act] = new_action_probability
        if self.second_algo_enable:
            # Calculate the expectation of the policy probabilities
            policy_expectation_delta = self.alpha*(policy - policy_expectation)
            policy_expectation += policy_expectation_delta
            # Using the expectation, calculate the second term of the algorithm: α*E[p(k)] − p(k), and add it to policy_delta
            policy += self.alpha*(policy_expectation - policy)
        if not (policy > 0).all():
            policy = self.softmax(policy)  # normalize probabilities
        return policy, policy_expectation

if __name__ == '__main__':
    k = 50000
    alpha = 0.001
    # PRISONERS DILEMMA 
    p1_pris = np.array([[ 5, 0], 
                        [10, 1]])
    p2_pris = p1_pris
    p1_policy = np.array([0.5,0.5])
    p2_policy = np.array([0.5,0.5])
    pris = PolicyIteration(p1_pris, p2_pris,p1_policy,p2_policy, alpha=alpha, k=k,
                           action_dict={0:"coop/lie to police",1:"defect/confess to police"},
                           second_algo_enable=False)
    p1_probs, p2_probs = pris.train()
    visualize_probabilities(p1_probs,p2_probs,k+2,"Prisoners Dilemma Probability Chart",
                            p1_labels=["P1 Cooperate","P1 Defect"],p2_labels=["P2 Cooperate","P2 Defect"])
    opposing_probabilities(p1_probs[:,0],p2_probs[:,0],"Probability of Choosing Cooperating")
    
    # HEADS AND TAILS
    p1_head_tails = np.array([[1, -1],
                              [-1, 1]])
    p2_head_tails = p1_head_tails*-1.
    p1_policy = np.array([0.2, 0.8])
    p2_policy = np.array([0.2, 0.8])
    heads_tails = PolicyIteration(p1_head_tails, p2_head_tails,p1_policy,p2_policy, alpha=alpha,k=k,
                                  action_dict={0:"showed heads",1:"showed tails"},
                                  second_algo_enable=False)
    p1_probs, p2_probs = heads_tails.train()
    visualize_probabilities(p1_probs,p2_probs,k+2,"Dual Probability of Choosing Heads",
                            p1_labels=["P1 Heads","P1 Tails"],p2_labels=["P2 Heads","P2 Tails"])
    opposing_probabilities(p1_probs[:,0],p2_probs[:,0],"Probability of Choosing Heads")

    # ROCK PAPER SCISSORS
    p1_rps    = np.array([[ 0,-1, 1],
                          [ 1, 0,-1],
                          [-1, 1, 0]])
    p2_rps    = p1_rps
    p1_policy = np.array([0.6, 0.2, 0.2])
    p2_policy = np.array([0.6, 0.2, 0.2])
    rps       = PolicyIteration(p1_rps, p2_rps, p1_policy, p2_policy, alpha=alpha, k=k,
                                action_dict={0: "threw rock", 1: "threw paper", 2: "threw scissors"},
                                second_algo_enable=False)
    p1_probs, p2_probs = rps.train()
    visualize_probabilities(p1_probs,p2_probs,k+2,"Rock Paper Scissors Probability Chart",
                            p1_labels=["P1 Rock","P1 Paper","P1 Scissors"],p2_labels=["P2 Rock","P2 Paper","P2 Scissors"])
    opposing_probabilities(p1_probs[:,0],p2_probs[:,0],"Probability of Choosing Rock")
    opposing_probabilities(p1_probs[:,1],p2_probs[:,1],"Probability of Choosing Paper")
