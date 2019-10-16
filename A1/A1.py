import numpy as np
class PolicyIteration(object):
    def __init__(self,p1_reward,p2_reward,alpha,k):
        super().__init__()
        self.p1_reward  = p1_reward
        self.p2_reward  = p2_reward
        self.actions    = [i for i in range(p1_reward.shape[0])]
        self.p1_policy  = np.random.dirichlet(np.ones(p1_reward.shape[0])) # Initialize random simplex for taking each action
        self.p2_policy  = np.random.dirichlet(np.ones(p1_reward.shape[0])) # Initialize random simplex for taking each action
        self.alpha      = alpha
        self.k          =  k
        self.train()
    
    def normalize_values(self,matrix):
        normalized = (matrix - np.min(matrix))/np.ptp(matrix)
        return normalized

    def softmax(self,x):
        """"
            Takes an n dimensional vector of real numbers, and normalizes it into a probability distribution 
            consisting of n probabilities proportional to the exponentials of the input numbers.
            https://en.wikipedia.org/wiki/Softmax_function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference

    def train(self):
        action_exp ={0: "showed heads",1:"showed tails"}
        for iteration in range(self.k):
            p1_action       = np.random.choice(self.actions,p=self.p1_policy)
            p2_action       = np.random.choice(self.actions,p=self.p2_policy)
            self.p1_policy = self.update_policy(self.p1_policy,self.p1_reward,p1_action,p2_action)
            self.p2_policy = self.update_policy(self.p2_policy,self.p2_reward,p1_action,p2_action)
            if self.k < 10 or iteration%(self.k//10) == 0:
                print("Iteration: {}".format(iteration))
                print("Player one {} and player two {}. Player one reward: {}. Player two reward: {}".format(action_exp[p1_action],action_exp[p2_action],self.p1_reward[p1_action,p2_action],self.p2_reward[p2_action,p1_action]))
                print("Probability of user one showing heads: {}, Probability of showing tails: {}".format(self.p1_policy[0],self.p1_policy[1]))
                print("Probability of user two showing heads: {}, Probability of showing tails: {}\n".format(self.p2_policy[0],self.p2_policy[1]))

    def update_policy(self,policy,reward,user_act,opponent_act):
        act_reward = reward[user_act,opponent_act]
        act_probability = policy[user_act]
        policy_delta    = self.alpha*act_reward*(1-act_probability)
        policy[user_act] += policy_delta
        for other_action in self.actions:
                if other_action != user_act:
                    act_probability = policy[other_action]
                    policy_delta = self.alpha*act_reward*(act_probability)
                    policy[other_action] -= policy_delta
        policy = self.softmax(policy) #normalize probabilities
        return policy




if __name__ == '__main__':
    p1_head_tails = np.array([[1,-1],[-1,1]])
    p1_head_tails = p1_head_tails
    p2_head_tails = -p1_head_tails
   
    test = PolicyIteration(p1_head_tails,p2_head_tails,alpha=0.5,k=10000)

