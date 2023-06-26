import numpy as np
import time
import matplotlib.pyplot as plt
from utilities import *
import torch
import torch.nn as nn
import torch.optim as optim
from agent import  Transition, DQN, Action, Memory
from gridWorld import GridWorld
from copy import deepcopy

BATCH_SIZE = 8
GAMMA = 0.99
TAU = 0.005
LR = 1e-3

num_episodes = 80
rewards = [0]
episode_reward = 0
memory = Memory(10_000)
plt.figure(1)

env = GridWorld(shape = (5,5), 
                init_agent_pos=(0,0),
                terminal_state=None,
                obstacles = [(0,1), (1,1), (2,1), (3,1),(2,3),(3,3),(4,3) ])
state, _ = env.reset()
policy_net = DQN(len(state), len(env.action_space))
target_net = deepcopy(policy_net) 
action_selector = Action(len(env.action_space), policy_net)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

def optimize_model():
  if len(memory) < BATCH_SIZE:
    return 
  transitions = memory.sample(BATCH_SIZE)
  batch = Transition(*zip(*transitions)) 
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
  non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
  state_batch = torch.cat(batch.state)
  # print(batch.state)
  # print(state_batch)
  # exit()
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  # print(action_batch)
  # print(policy_net(state_batch))
  # exit()
  state_action_values = policy_net(state_batch).gather(1, action_batch)

  next_state_values = torch.zeros(BATCH_SIZE)
  with torch.no_grad():
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
  expected_state_action_values = ((next_state_values * GAMMA) + reward_batch).unsqueeze(1)
  # print(state_action_values.shape)
  # print(expected_state_action_values.shape)
  # exit()

  criterion = nn.SmoothL1Loss()
  loss = criterion(state_action_values, expected_state_action_values)
  optimizer.zero_grad()
  loss.backward()
  torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
  optimizer.step()

for i_episode in range(num_episodes):
  state, _ = env.reset()
  state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
  while True:
    env.render()
    action = action_selector.select(state)
    observation, reward, done, _ = env.step(action.item())
    episode_reward += reward

    reward = torch.tensor([reward])

    if done:
      next_state = None
    else:
      next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    # print( state, action, next_state, reward)
    # exit()
    memory.push(state, action, next_state, reward)
    state = next_state
    optimize_model()
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
      target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
      (1-TAU) * target_net_state_dict[key]
    target_net.load_state_dict(target_net_state_dict)

    print(f"Progress: {i_episode+1}/{num_episodes}")
        
    if done:
        rewards.append(episode_reward)
        episode_reward = 0
        plot(rewards)
        print("DONE!")
        torch.save(policy_net, "policy_net.model")
        env.render()
        break

plt.savefig("training_results.pdf")