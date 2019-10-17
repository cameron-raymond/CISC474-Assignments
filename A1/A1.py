import numpy as np
from visualizations import visualize_probabilities


class PolicyIteration(object):
    def __init__(self, p1_reward, p2_reward, alpha, k, action_dict=None,intermittent_normalization=True):
        super().__init__()
        self.p1_reward = p1_reward
        self.p2_reward = p2_reward
        self.intermittent_normalization = intermittent_normalization
        self.actions = [i for i in range(p1_reward.shape[0])]
        # Initialize random simplex for taking each action
        self.p1_policy = np.random.dirichlet(np.ones(p1_reward.shape[0]))
        # Initialize random simplex for taking each action
        self.p2_policy = np.random.dirichlet(np.ones(p1_reward.shape[0]))
        self.alpha = alpha
        self.k = k
        self.action_dict = action_dict

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
                # For some problems, it helped to normalize the probabilities every once in a while
                if self.intermittent_normalization:
                    self.p1_policy = self.softmax(self.p1_policy)
                    self.p2_policy = self.softmax(self.p2_policy)
                print("Iteration: {}".format(iteration+1))
                print("Player one {} and player two {}.".format(action_dict[p1_action], action_dict[p2_action]))
                print("Player one reward: {}. Player two reward: {}".format(self.p1_reward[p1_action, p2_action], self.p2_reward[p1_action,p2_action]))
                print("Player one policy {}".format(self.p1_policy))
                print("Player two policy {}".format(self.p2_policy))
            self.p1_policy = self.update_policy(self.p1_policy, self.p1_reward, p1_action, p2_action)
            self.p2_policy = self.update_policy(self.p2_policy, self.p2_reward, p1_action, p2_action)
            p1_over_time = np.vstack([p1_over_time,self.p1_policy])
            p2_over_time = np.vstack([p2_over_time,self.p2_policy])
        return p1_over_time, p2_over_time

    def update_policy(self, policy, reward, user_act, opponent_act):
        # reward for an action also depends on what the other user did
        act_reward = reward[user_act, opponent_act]
        act_probability = policy[user_act]
        # from equation 𝑝(𝑘 + 1) = 𝑝(𝑘) + 𝛼𝑟(𝑘)(1 − 𝑝(𝑘))
        action_delta = self.alpha*act_reward*(1-act_probability)
        # update all other probabilities 𝑝(𝑘 + 1) = 𝑝(𝑘) − 𝛼𝑟(𝑘)*𝑝(𝑘) = 𝑝(𝑘) + −1*𝛼𝑟(𝑘)*𝑝(𝑘), 𝑓𝑜𝑟 𝑎𝑙𝑙 𝑜𝑡h𝑒𝑟 𝑎𝑐𝑡𝑖𝑜𝑛𝑠 𝑜 ≠ 𝑐
        # this is done by multiplying the policy vector by scalars alpha, the action reward and -1
        policy_delta = -1.*self.alpha*act_reward*policy
        # bring in the delta from the action we actually chose
        policy_delta[user_act] = action_delta
        policy += policy_delta
        if not (policy > 0).all():
            policy = self.softmax(policy)  # normalize probabilities
        return policy

if __name__ == '__main__':
    k=50000
    # HEADS AND TAILS
    p1_head_tails = np.array([  [1, -1], 
                                [-1, 1]])
    p2_head_tails = p1_head_tails*-1.
    heads_tails = PolicyIteration(p1_head_tails, p2_head_tails, alpha=0.001, k=k,action_dict={0:"showed heads",1:"showed tails"})
    # p1_probs, p2_probs = heads_tails.train()
    # visualize_probabilities(p1_probs,p2_probs,k+2,"Dual Probability of Choosing Heads",p1_labels=["P1 Heads","P1 Tails"],p2_labels=["P2 Heads","P2 Tails"])
   
    # ROCK PAPER SCISSORS 
    p1_rps = np.array([ [0, -1, 1], 
                        [1, 0, -1], 
                        [-1, 1, 0]])
    p2_rps = -p1_rps
    rps = PolicyIteration(p1_rps, p2_rps, alpha=0.001, k=k,action_dict={0:"threw rock",1:"threw paper",2:"threw scissors"})
    # p1_probs, p2_probs = rps.train()
    # visualize_probabilities(p1_probs,p2_probs,k+2,"Rock Paper Scissors Probability Chart",p1_labels=["P1 Rock","P1 Paper","P1 Scissors"],p2_labels=["P2 Rock","P2 Paper","P2 Scissors"])

    # PRISONERS DILEMMA 
    p1_pris = np.array([[ 5, 0], 
                        [10, 1]])
    p2_pris = p1_pris.T
    pris = PolicyIteration(p1_pris, p2_pris, alpha=0.001, k=k,action_dict={0:"coop/lie to police",1:"defect/confess to police"},intermittent_normalization=False)
    p1_probs, p2_probs = pris.train()
    visualize_probabilities(p1_probs,p2_probs,k+2,"Prisoners Dilemma Probability Chart",p1_labels=["P1 Cooperate","P1 Defect"],p2_labels=["P2 Cooperate","P2 Defect"])

