"""
    Main module for running the simulations.
"""

from Q_Learning  import Q_Learning
from SARSA       import SARSA

# Defining learning parameters
episodes = 300
learning_rate = 0.5
discount = 0.9
epsilon = 0.05


if __name__ == "__main__":
    # Performing Q-learning with standard environment
    q_agent = Q_Learning(episodes=episodes,lr=learning_rate,discount=discount,epsilon=epsilon,king=False)
    print("--- starting training for Q Learning agent --- ")
    episode_steps = q_agent.train()
    print("--- plotting training for Q Learning agent ---")
    q_agent.plot(episode_steps)

    # Performing SARSA with standard environment
    sarsa_agent = SARSA(episodes=episodes,lr=learning_rate,discount=discount,epsilon=epsilon,king=False)
    print("--- starting training for SARSA agent --- ")
    episode_steps = sarsa_agent.train()
    print("--- plotting training for SARSA agent ---")
    sarsa_agent.plot(episode_steps)

    # Performing Q-learning with king's moves and stochastic wind
    q_agent_king = Q_Learning(episodes=episodes,lr=learning_rate,discount=discount,epsilon=epsilon,king=True)
    print("--- starting training for Q Learning agent with king's move --- ")
    episode_steps = q_agent_king.train()
    print("--- plotting training for Q Learning agent ---")
    q_agent_king.plot(episode_steps)

    # Performing SARSA with king's moves and stochastic wind
    sarsa_agent_king = SARSA(episodes=episodes,lr=learning_rate,discount=discount,epsilon=epsilon,king=True)
    print("--- starting training for SARSA agent with king's move --- ")
    episode_steps = sarsa_agent_king.train()
    print("--- plotting training for SARSA agent with king's move ---")
    sarsa_agent_king.plot(episode_steps)
