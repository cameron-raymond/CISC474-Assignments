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
