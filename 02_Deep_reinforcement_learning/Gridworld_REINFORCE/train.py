# vanilla policy gradient 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np
from collections import deque
from collections import namedtuple 

import sys
import os
sys.path.append( # add parent directory to the python path so you can import from sub directories
    os.path.dirname( # get the name of the parent directory
    os.path.dirname( # get the name of the parent directory
    os.path.abspath(__file__) # absolute path of this file
    )))
import matplotlib.pyplot as plt
from utils.utilities import *
from agent import Policy
from gridWorld import GridWorld  # Import GridWorld environment

LR = 1e-3
GAMMA = 0.99
EPISODES = 3000
RENDER = True 
RENDER_EVERY = 50 
HIDDEN_SIZE = 128

Transitions = namedtuple('Transitions', ('states', 'actions', 'rewards'))




def reward_to_go(rewards, gamma):
    """Calculate rewards to go for a given episode (i.e., the rewards after taking an action): 
        # G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^(T) * r_T
        # where T is the last time step of the episode
    Args:
        rewards (list-like): list of rewards for the episode
        gamma (float): discount factor
    Returns:
        list (or list-like): list of rewards to go
    """
    R=0
    returns = []
    for r in rewards[::-1]:  # reverse the list
        R = r + gamma * R # cumulative rewards for each time step
        returns.insert(0, R) # insert at the beginning of the list
    return returns 

def optimize(transitions):
    """Optimize the policy network using the policy gradient algorithm
    Args:  
        transitions (namedtuple): namedtuple containing the states, actions, and rewards for an episode
    """
    rewards_to_go = torch.tensor(reward_to_go(transitions.rewards, GAMMA))
    states = torch.tensor(transitions.states).float()
    actions = torch.tensor(transitions.actions)

    # print(f'states: {states}')
    probs = policy(states)
    
    ## convert the logits to log probabilities; negative sign to do gradient ascent:
    #Categorical(probs) creates a categorical distribution object based on the given 
    # probability values probs. This object represents a discrete probability distribution 
    # where each action has a corresponding probability.log_prob(actions) calculates the 
    # natural logarithm of the probability of each action in the actions set, according to 
    # the categorical distribution. This operation returns a tensor or an array of logarithmic 
    # probabilities. The negative sign - negates each logarithmic probability in the tensor or 
    # array, resulting in negative log probabilities. 
    log_probs = - Categorical(probs).log_prob(actions) 
    loss = torch.sum(log_probs * rewards_to_go) # loss is the sum of the log probabilities times the rewards_to_go
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# if __name__ == "__main__":

# Initialize the GridWorld environment
env = GridWorld(shape = (5,5), 
                init_agent_pos=(0,0),
                terminal_pos=None,
                obstacles = [(0,1), (1,1), (2,1), (3,1),(2,3),(3,3),(4,3) ])
state, _ = env.reset()  # Reset the environment

policy = Policy(len(state), len(env.action_space), HIDDEN_SIZE)
optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

for episode in range(EPISODES):
    done = False
    transitions = Transitions(states=[], actions=[], rewards=[])
    state, _ = env.reset()

    while True:
        if RENDER and episode % RENDER_EVERY == 0:
            env.render()
        state_tensor = torch.tensor(state).unsqueeze(0).float()
        # get action probabilities from the policy network
        action_probs = policy(state_tensor)
        # sample action from the action_probs distribution
        # action is the index of the action in the action space 
        action = Categorical(action_probs).sample().item()
   
        # t = env.step(action) ->  (array([ 0.04733234,  0.17882274,  0.04574946, -0.29644156], dtype=float32), 1.0, False, False, {})
        next_state, reward, done, _, = env.step(action)
        transitions.states.append(state)
        transitions.actions.append(action)
        transitions.rewards.append(reward)
        state = next_state

        if done: 
            break
    
    optimize(transitions)
    print(f'Episode {episode+1} return: {sum(transitions.rewards)}')
    torch.save(policy, "results/REINFORCE.model")  # Save the trained model

env.close(info= transitions.rewards)        

        


