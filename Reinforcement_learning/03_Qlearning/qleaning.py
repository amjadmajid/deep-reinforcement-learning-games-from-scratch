# Import necessary libraries
import numpy as np
import time
import sys
import os

# Add parent directory to the Python path to enable importing from subdirectories
sys.path.append(
    os.path.dirname(
    os.path.dirname(
    os.path.abspath(__file__)  # Absolute path of this file
    )))

from Games.utils import *  # Import utility functions
from Games.gridWorld import GridWorld  # Import GridWorld environment

# Define QLearning class
class QLearning():
    def __init__(self, env, episodes=10000, epsilon=.2, alpha=.1, gamma=.99):
        # Initialize action values, episodes, learning rate, discount factor, and exploration rate
        self.action_values = np.zeros((env.rows, env.cols, env.action.action_n))
        self.episodes =  episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # Define exploratory policy for taking random actions
    def exploratory_policy(self, state):
        return np.random.choice(env.action.action_idxs) 
        
    # Define target policy for taking the action with maximum expected future reward
    def target_policy(self, state):
        av = self.action_values [state[0]][state[1]]
        return np.random.choice(np.flatnonzero(av == av.max()))

    # Define Q-learning function
    def qlearning(self):
        for _ in range(1, self.episodes+1):  # For each episode
            state = env.reset()  # Reset the environment
            done = False  # Initialize 'done' flag

            while not done:  # While episode is not done
                action = self.exploratory_policy(state)  # Select action according to exploratory policy
                next_state, reward, done, _ = env.step(action)  # Perform action and get next state, reward, and 'done' flag
                next_action = self.target_policy(next_state)  # Select action according to target policy for the next state

                # Calculate and update Q-value for the state-action pair
                qsa = self.action_values[state[0]][state[1]][action] 
                next_qsa = self.action_values[next_state[0]][next_state[1]][next_action] 
                self.action_values[state[0]][state[1]][action] = qsa + self.alpha *(reward + self.gamma * next_qsa - qsa)

                state = next_state  # Move to next state

# Main script
if __name__ == '__main__':
    # Define the GridWorld environment
    env = GridWorld(shape = (5,5), obstacles = [(0,1), (1,1), (2,1), (4,1),\
                    (0,3),(2,3),(3,3), (4,3) ])
    
    # Initialize and run Q-learning
    qlearning = QLearning(env, episodes = 1000, epsilon=.2)
    qlearning.qlearning()
    
    steps =0  # Initialize step counter
    done = False  # Initialize 'done' flag
    env.reset()  # Reset environment

    while True:  # Loop until agent reaches terminal state
        steps += 1  # Increment step counter
        clear_screen()  # Clear console output
        state = env.get_agent_pos()  # Get agent's current position
        action = qlearning.target_policy(state)  # Select action according to target policy
        state, _, done, _ = env.step(action)  # Perform action and get next state and 'done' flag
        env.render()  # Render environment

        if done:  # If agent reached terminal state
            print(f"the agent reached terminal state in {steps} steps")  # Print number of steps taken
            plot_q_table(qlearning.action_values)  # Plot Q-table
            break  # Break loop

        time.sleep(.5)  # Pause for 0.5 seconds before next step
