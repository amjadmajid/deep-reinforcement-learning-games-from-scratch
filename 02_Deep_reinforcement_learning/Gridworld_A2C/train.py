# vanilla policy gradient 
import torch
import torch.optim as optim
from torch.distributions import Categorical
import gym
from collections import namedtuple 
from itertools import count

import sys
import os
sys.path.append( # add parent directory to the python path so you can import from sub directories
    os.path.dirname( # get the name of the parent directory
    os.path.dirname( # get the name of the parent directory
    os.path.abspath(__file__) # absolute path of this file
    )))
import matplotlib.pyplot as plt
# from utils.utilities import plot_durations, episode_durations
from agent import Actor, Critic
from gridWorld import GridWorld  # Import GridWorld environment
from utils.utilities import plot_durations, episode_durations

LR = 1e-3
GAMMA = 0.99
EPISODES = 3000
RENDER = True 
RENDER_EVERY = 50 
HIDDEN_SIZE = 128

Transitions = namedtuple('Transitions', ('states', 'actions', 'rewards'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GridWorld(shape = (5,5), 
                init_agent_pos=(0,0),
                terminal_pos=None,
                obstacles = [(0,1), (1,1), (2,1), (3,1),(2,3),(3,3),(4,3) ])
state, _ = env.reset()  # Reset the environment

state_size = len(state)
action_size = len(env.action_space)
lr = 0.0001

def compute_returns(rewards, gamma=0.99):
    R = 0 
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R  
        returns.insert(0, R)
    return returns

def play_episode(actor, critic):
    state, _ = env.reset()
    log_probs = []; values = []; rewards = []
    entropy = 0

    for i in count():
        # env.render()
        state = torch.FloatTensor(state).to(device)
        dist, value = actor(state), critic(state)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        log_prob = dist.log_prob(action).unsqueeze(0)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
        
        state = next_state

        if done:
            torch.save(actor, "results/A2C.model")
            episode_durations.append(i + 1)
            plot_durations() 
            break

    return log_probs, values, rewards, entropy

def optimize_model(optimizerA, optimizerC, log_probs, values, rewards, entropy, entropy_coef=0):
    returns = compute_returns(rewards)
    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values
    actor_loss = -(log_probs * advantage.detach()).mean() - entropy_coef * entropy
    critic_loss = advantage.pow(2).mean()

    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()

    return actor_loss.item(), critic_loss.item()

            
def trainIters(actor, critic, n_iters, entropy_coef=0, log_interval=10):
    for episode in range(n_iters):
        log_probs, values, rewards, entropy = play_episode(actor, critic)
        actor_loss, critic_loss = optimize_model(optimizerA, optimizerC, log_probs, values, rewards, entropy)


if __name__ == '__main__':

    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    optimizerA = optim.AdamW(actor.parameters(),  lr=lr)
    optimizerC = optim.AdamW(critic.parameters(), lr=lr)
    trainIters(actor, critic, n_iters=500)

