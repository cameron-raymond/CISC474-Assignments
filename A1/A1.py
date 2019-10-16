import numpy as np

class PolicyIteration(object):
    def __init__(self, p1_reward, p2_reward, alpha, k,action_dict=None):
        super().__init__()
        self.p1_reward = p1_reward
        self.p2_reward = p2_reward
        self.actions = [i for i in range(p1_reward.shape[0])]
        # Initialize random simplex for taking each action
        self.p1_policy = np.random.dirichlet(np.ones(p1_reward.shape[0]))
        # Initialize random simplex for taking each action
        self.p2_policy = np.random.dirichlet(np.ones(p1_reward.shape[0]))
        self.alpha = alpha
        self.k = k
        self.train(action_dict)

    def softmax(self, x):
        """"
            Takes an n dimensional vector of real numbers, and normalizes it into a probability distribution 
            consisting of n probabilities proportional to the exponentials of the input numbers.
            https://en.wikipedia.org/wiki/Softmax_function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def train(self,action_dict):
        for iteration in range(self.k):
            p1_action = np.random.choice(self.actions, p=self.p1_policy)
            p2_action = np.random.choice(self.actions, p=self.p2_policy)
            self.p1_policy = self.update_policy(self.p1_policy, self.p1_reward, p1_action, p2_action)
            self.p2_policy = self.update_policy(self.p2_policy, self.p2_reward, p1_action, p2_action)
            if self.k < 10 or iteration % (self.k//10) == 0:
                print("Iteration: {}".format(iteration))
                print("Player one {} and player two {}. Player one reward: {}. Player two reward: {}".format(
                    action_dict[p1_action], action_dict[p2_action], self.p1_reward[p1_action, p2_action], self.p2_reward[p2_action, p1_action]))
                print("Player one policy {}".format(self.p1_policy))
                print("Player two policy {}".format(self.p2_policy))


    def update_policy(self, policy, reward, user_act, opponent_act):
        # reward for an action also depends on what the other user did
        act_reward = reward[user_act, opponent_act]
        act_probability = policy[user_act]
        # from equation 𝑝(𝑘 + 1) = 𝑝(𝑘) + 𝛼𝑟(𝑘)(1 − 𝑝(𝑘))
        action_delta = self.alpha*act_reward*(1-act_probability)
        # update all other probabilities 𝑝(𝑘 + 1) = 𝑝(𝑘) − 𝛼𝑟(𝑘)*𝑝(𝑘) = 𝑝(𝑘) + −1*𝛼𝑟(𝑘)*𝑝(𝑘), 𝑓𝑜𝑟 𝑎𝑙𝑙 𝑜𝑡h𝑒𝑟 𝑎𝑐𝑡𝑖𝑜𝑛𝑠 𝑜 ≠ 𝑐
        policy_delta = -1.*self.alpha*act_reward*policy
        # bring in the delta from the action we actually chose
        policy_delta[user_act] = action_delta
        policy += policy_delta
        policy  = self.softmax(policy)  # normalize probabilities
        return policy



if __name__ == '__main__':
    p1_head_tails = np.array([[1, -1], [-1, 1]])
    p2_head_tails = -p1_head_tails

    heads_tails = PolicyIteration(p1_head_tails, p2_head_tails, alpha=0.001, k=50000,action_dict={0:"showed heads",1:"showed tails"})

    # p1_rps = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    # p2_rps = -p1_rps
    # rps = PolicyIteration(p1_rps, p2_rps, alpha=0.001, k=50000,action_dict={0:"threw rock",1:"threw paper",2:"threw scissors"})
    # p1_prisoners = np.array([[5, 0], [10, 1]])
    # p2_prisoners = np.transpose(p1_prisoners)
    # prisoners = PolicyIteration(p1_prisoners,p2_prisoners,alpha=0.001,k=50000,action_dict={0:"cooperates/lies",1:"confesses"})