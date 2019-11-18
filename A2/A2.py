from Q_Learning  import Q_Learning
from SARSA       import SARSA

episodes = 300
learning_rate = 0.5
discount = 0.9
epsilon = 0.05

if __name__ == "__main__":
    q_agent = Q_Learning(episodes=episodes,lr=learning_rate,discount=discount,epsilon=epsilon)
    print("--- starting training for Q Learning agent --- ")
    episode_steps = q_agent.train()
    print("--- plotting trainging for Q Learning agent ---")
    q_agent.plot(episode_steps)
    sarsa_agent = SARSA(episodes=episodes,lr=learning_rate,discount=discount,epsilon=epsilon)
    print("--- starting training for SARSA agent --- ")
    episode_steps = sarsa_agent.train()
    print("--- plotting trainging for SARSA agent ---")
    sarsa_agent.plot(episode_steps)

