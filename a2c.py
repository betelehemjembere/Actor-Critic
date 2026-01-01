                #group member name and ID
            # Betelehem Jembere – UGR/3643/15
            # Dawit Genene – UGR/0905/15
            # Kirubel Tesfaye – UGR/9575/15


import numpy as np
import random

# Number of states and actions
states = 10
actions = 2

# Initialize Actor (policy) and Critic (value function)
policy = np.random.rand(states, actions)
value = np.random.rand(states)

# Learning parameters
alpha_actor = 0.01
alpha_critic = 0.05
gamma = 0.9
episodes = 300

total_rewards = []

for episode in range(episodes):
    state = random.randint(0, states - 1)
    episode_reward = 0

    # Select action using Actor (greedy)
    action = np.argmax(policy[state])

    # Simulated environment response
    reward = random.choice([0, 1])
    next_state = random.randint(0, states - 1)

    # Temporal Difference error
    td_error = reward + gamma * value[next_state] - value[state]

    # Critic update
    value[state] += alpha_critic * td_error

    # Actor update
    policy[state, action] += alpha_actor * td_error

    episode_reward += reward
    total_rewards.append(episode_reward)

print("Training completed successfully.")
print("Average reward:", np.mean(total_rewards))
print("\nFinal Value Function:")
print(value)
print("\nFinal Policy Table:")
print(policy)
