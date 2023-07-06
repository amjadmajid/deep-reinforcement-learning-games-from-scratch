# Import necessary libraries
import numpy as np
import time
import sys
import os

sys.path.append( # add parent directory to the python path so you can import from sub directories
    os.path.dirname( # get the name of the parent directory
    os.path.abspath(__file__) # absolute path of this file
    ))
import matplotlib.pyplot as plt
from utils.utilities import *
import torch
import torch.nn as nn
import torch.optim as optim
from agent import Transition, DQN, Action, Memory  # Import classes from the agent module
from gridWorld import GridWorld  # Import GridWorld environment
from copy import deepcopy

# Define hyperparameters
BATCH_SIZE = 8
GAMMA = 0.99
TAU = 0.005
LR = 1e-3

num_episodes = 80
rewards = [0]
episode_reward = 0
memory = Memory(10_000)  # Initialize a Memory object with a capacity of 10,000
plt.figure(1)

# Initialize the GridWorld environment
env = GridWorld(shape = (5,5), 
                init_agent_pos=(0,0),
                terminal_pos=None,
                obstacles = [(0,1), (1,1), (2,1), (3,1),(2,3),(3,3),(4,3) ])

state, _ = env.reset()  # Reset the environment

# Initialize the policy network and the target network
policy_net = DQN(len(state), len(env.action_space))
target_net = deepcopy(policy_net) 

# Initialize the action selector
action_selector = Action(len(env.action_space), policy_net)

# Initialize the optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Function to optimize the model
def optimize_model():
  if len(memory) < BATCH_SIZE:  # If the memory is not yet filled up to the batch size, return
    return 
  transitions = memory.sample(BATCH_SIZE)  # Sample a batch of transitions from memory
  batch = Transition(*zip(*transitions))  # Unzip the transitions to create a batch
  
  # Compute a mask of non-final states
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))

  # Concatenate all the non-final next states
  non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  state_action_values = policy_net(state_batch).gather(1, action_batch)

  # Compute next state values for all non-final states, initialize with zeros
  next_state_values = torch.zeros(BATCH_SIZE)

  # Compute next state values for non-final states
  with torch.no_grad():
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

  # Compute the expected state-action values
  expected_state_action_values = ((next_state_values * GAMMA) + reward_batch).unsqueeze(1)

  # Compute the loss
  criterion = nn.SmoothL1Loss()
  loss = criterion(state_action_values, expected_state_action_values)

  # Optimize the model
  optimizer.zero_grad()
  loss.backward()
  torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # Clip gradients to stabilize training
  optimizer.step()

# Main training loop
for i_episode in range(num_episodes):
  state, _ = env.reset()
  state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
  while True:
    env.render()  # Render the environment
    action = action_selector.select(state)  # Select an action
    observation, reward, done, _ = env.step(action.item())  # Perform the action
    episode_reward += reward

    reward = torch.tensor([reward])

    # If the episode has ended, then the next state is None
    if done:
      next_state = None
    else:
      next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
 
    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    state = next_state
    optimize_model()  # Optimize the model

    # Soft update the target network
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
      target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
      (1-TAU) * target_net_state_dict[key]
    target_net.load_state_dict(target_net_state_dict)

    print(f"Progress: {i_episode+1}/{num_episodes}")
        
    if done:
        rewards.append(episode_reward)  # Save the episode reward
        episode_reward = 0
        plot(rewards)  # Plot the rewards
        print("DONE!")
        torch.save(policy_net, "results/DQN.model")  # Save the trained model
        env.render()  # Render the final state of the environment
        break

# Save the training results
plt.savefig("results/training_results.pdf")
