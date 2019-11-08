import matplotlib.pyplot as plt
import numpy as np
from lib.envs.windy_gridworld import WindyGridworldEnv
from collections import defaultdict

def epsilon_greedy_policy(Q, state, nA, epsilon):
    '''
    Returns greedy policy with epsilon chance of a random action
    '''
    probs = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q[state])
    probs[best_action] += 1.0 - epsilon

    return probs

def Q_learning(episodes, learning_rate, discount, epsilon):
    '''
    Learn to solve the environment using Q-learning

    :episodes: Number of episodes to run (int)
    :param lr: learning rate (float [0, 1])
    :param discount: alpha discount factor (float [0, 1])
    :param epsilon: chance a random move is selected (float [0, 1])
    :return: x,y points to graph
    '''

    # Links state to action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Points to plot
    # number of episodes
    x = np.arange(episodes)
    # Number of steps
    y = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()

        for step in range(10000):

            # Select and take action
            probs = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)

            # Update
            Q[state][action] +=  lr * (reward + (discount * np.amax(Q[next_state])) - Q[state][action]) 

            if done:
                y[episode] = step
                break

            state = next_state

    return x, y


# Define environment and learning params 
env = WindyGridworldEnv()
episodes = 100
learning_rate = 0.5
discount = 0.9
epsilon = 0.05
_lambda = 0.9

# Learn
x, y = Q_learning(episodes, learning_rate, discount, epsilon)

# Show results
_, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='Episodes', ylabel='steps',
       title='Episodes vs steps')
ax.grid()

plt.show()
