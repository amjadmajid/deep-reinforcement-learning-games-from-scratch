# Import necessary libraries
import numpy as np
import time
import sys
import os

sys.path.append( # add parent directory to the python path so you can import from sub directories
    os.path.dirname( # get the name of the parent directory
    os.path.dirname( # get the name of current directory
    os.path.abspath(__file__) # absolute path of this file
    )))

from Games.utils import * 
from Games.gridWorld import GridWorld

# Define Sarsa class
class Sarsa():
    # Initialization function
    def __init__(self, env, episodes=10000, epsilon=.2, alpha=.1, gamma=.99):
        # Initialize action-value function (Q function) to zeros
        self.action_values = np.zeros((env.rows, env.cols, env.action.action_n))
        self.episodes =  episodes  # Total number of episodes to train
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy

    # Function to determine which action to take
    def policy(self, state):
        # Randomly choose an action with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.choice(env.action.action_idxs) 
        else:  # Choose best action with probability 1-epsilon
            av = self.action_values [state[0]][state[1]]
            return np.random.choice(np.flatnonzero(av == av.max()))  # Break ties randomly

    # SARSA algorithm
    def sarsa(self):
        # Iterate over episodes
        for _ in range(1, self.episodes+1):
            # Reset state at the start of each episode
            state = env.reset()
            # Choose action based on current policy
            action = self.policy(state)
            done = False

            # Loop until the end of episode
            while not done:
                # Take action and get result
                next_state, reward, done, _ = env.step(action)
                # Choose next action based on current policy
                next_action = self.policy(state)

                # Update action-value function using the SARSA update rule
                qsa = self.action_values[state[0]][state[1]][action] 
                next_qsa = self.action_values[next_state[0]][next_state[1]][next_action] 
                self.action_values[state[0]][state[1]][action] = qsa + self.alpha *(reward + self.gamma * next_qsa - qsa)

                # Move to the next state
                state = next_state
                # Next action becomes the current action
                action = next_action

# Main program
if __name__ == '__main__':
    # Define the GridWorld environment
    env = GridWorld(shape = (5,5), obstacles = [(0,1), (1,1), (2,1), (4,1),\
                    (0,3),(2,3),(3,3), (4,3) ])
    
    # Initialize Sarsa algorithm
    sarsa = Sarsa(env, episodes = 10000, epsilon=.2)
    # Train the agent
    sarsa.sarsa()
    steps =0
    done = False
    env.reset()
    # Test the agent
    while True: 
        steps += 1
        clear_screen()
        state = env.get_agent_pos()
        action = sarsa.policy(state)
        state, _, done, _ = env.step(action)
        env.render()
        if done:
            print(f"the agent reached terminal state in {steps} steps")
            plot_q_table(sarsa.action_values)
            break
        time.sleep(.5)
