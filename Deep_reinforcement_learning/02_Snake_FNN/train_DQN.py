import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from agent import  Transition, DQN, Action, Memory
from snake import Snake
from env import Grid, Window
from copy import deepcopy
from utilities import plot
import pygame
import time

BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.05
LR = 1e-3

num_episodes = 10_000
rewards = []
memory = Memory(10_0000)
plt.figure(1)

grid = Grid()
snake = Snake(grid)
window = Window(grid, snake)
action_space_size = len(snake.direction.directions)-1 # do not allow 180 degree move
state, _ = snake.reset()

# print(f"STATE {state}, ACTION_SPACE_SIZE  {action_space_size}")

# exit()

policy_net = DQN(len(state), action_space_size)
target_net = deepcopy(policy_net) 

action_selector = Action(action_space_size, policy_net)
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
  # print(action_batch.argmax(1))
  # print(action_batch)
  # print(policy_net(state_batch))
  # exit()
  state_action_values = policy_net(state_batch).gather(1, action_batch.argmax(1).unsqueeze(1))
  # print(state_action_values)
  # exit()
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

playing = True
def _handle_user_input():
    '''This method handles user input'''
    for event in pygame.event.get():
        # pygame.QUIT event happens when the user click on the window closing button 
        if event.type == pygame.QUIT:
            playing = False
            pygame.quit()   # quit pygame

def update_target_net():
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
      target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
      (1-TAU) * target_net_state_dict[key]
    target_net.load_state_dict(target_net_state_dict)
   
for i_episode in range(num_episodes):
  state, _ = snake.reset()
  state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
  episode_reward = 0
  
  while playing:
    window.update()
    action = action_selector.select(state)
    observation, reward, done, _ = snake.step(action)
    episode_reward += reward
    reward = torch.tensor([reward])

    if done:
      next_state = None
    else:
      next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    print( state, action, next_state, reward)
    if reward == 10 :
      time.sleep(.2)
    memory.push(state, action, next_state, reward)
    state = next_state
    optimize_model()
    update_target_net()
    _handle_user_input()  
    
    print(f"Progress: {i_episode+1}/{num_episodes}")

    if done:
        rewards.append(episode_reward)

        plot(rewards)
        print("DONE!")
        torch.save(policy_net, "DQN.model")
        window.update()
        break
    

plt.savefig("training_results.pdf")