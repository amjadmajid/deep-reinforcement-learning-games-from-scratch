import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from agent import Transition, DQN, Action, Memory
from snake import Snake
from env import Grid, Window
from copy import deepcopy
import sys
import os

sys.path.append( # add parent directory to the python path so you can import from sub directories
    os.path.dirname( # get the name of the parent directory
    os.path.dirname( # get the name of the parent directory
    os.path.abspath(__file__) # absolute path of this file
    )))
from utils.utilities import plot
import pygame
import time
import numpy as np
from collections import deque

MODEL = 'DQN'
# MODEL = 'DDQN'

BATCH_SIZE = 16
GAMMA = 0.97
TAU = 0.01
LR = 1e-3
FRAME_STACK_SIZE = 2
grid_size = (20, 24)  # Adjust the grid size to match your game's layout size
cell_size = 16
num_episodes = 5_000
rewards = []
memory = Memory(10_0000)
# plt.figure(1)

class FrameStacker():
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frames = deque([], maxlen=self.stack_size)
    
    def push(self, frame):
        self.frames.append(frame.squeeze(0))
        
    def get_stacked_frames(self):
        # If the stack is not filled yet, fill all with the current frame
        while len(self.frames) < self.stack_size:
            self.frames.append(self.frames[-1])
        # Stack frames
        stacked_frames = np.stack(self.frames, axis=0)
        # print('[TRAINING: STACKED FRAMES]', stacked_frames.shape)
        return stacked_frames
    
    def clear(self):
        self.frames.clear()

frame_stacker = FrameStacker(stack_size=FRAME_STACK_SIZE)

grid = Grid(grid_size=grid_size, cell_size=cell_size)
snake = Snake(grid)

# window = Window(grid, snake)
action_space_size =  3 #len(snake.direction.directions) - 1  # Do not allow 180 degree move

policy_net = DQN((FRAME_STACK_SIZE, *grid.grid.shape), action_space_size)  # Change input channels to 4 because of frame stacking
target_net = deepcopy(policy_net)

action_selector = Action(policy_net, action_space_size)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)



def optimize_model(Model=MODEL):
    if len(memory) < BATCH_SIZE * 2:
        return
    # print('[TRAINING: OPTIMIZE MODEL]')
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    # print('[TRAINING: STATE BATCH]', state_batch.shape)
    action_batch = torch.cat(batch.action)
    # print('[TRAINING: ACTION BATCH]', action_batch)
    reward_batch = torch.cat(batch.reward)
    next_state_stack = torch.cat(batch.next_state)
    # print('[TRAINING: NEXT STATE]', next_state_stack.shape)
    # print('[DONE BATCH ]', batch.done)
    done_batch = torch.cat(batch.done)

    # print('[TRAINING: STATE BATCH]', state_batch.shape)
    policy_net_out = policy_net(state_batch)
    # print('[TRAINING: POLICY NET OUT]', policy_net_out.shape, policy_net_out)
    state_action_values = policy_net_out.gather(1, action_batch.argmax(1).unsqueeze(1))

    # print('[TRAINING: STATE ACTION VALUES]', state_action_values.shape, state_action_values)
    # exit()
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        if Model == 'DQN':
            next_state_values = target_net(next_state_stack).max(1)[0]
        elif Model == 'DDQN':
            max_Q_action = policy_net(next_state_stack).max(1)[1].unsqueeze(1)
            # print('[MAX Q ACTION]', max_Q_action.shape, max_Q_action)
            next_state_values = target_net(next_state_stack).gather(1, max_Q_action)
        next_state_values[done_batch] = 0.0
    expected_state_action_values = ((next_state_values * GAMMA) + reward_batch).unsqueeze(1)


    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    # print('[LOSS]', loss)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

playing = True

def _handle_user_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global playing
            playing = False
            pygame.quit()

def hard_update_target_net():
    target_net.load_state_dict(policy_net.state_dict())

def update_target_net():
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
            (1-TAU) * target_net_state_dict[key]
    target_net.load_state_dict(target_net_state_dict)

for i_episode in range(num_episodes):
    print("STARTING EPISODE: ", i_episode)
    snake_length = len(snake.snake)
    
    ## Reset behavior
    frame_stacker.clear()
    state, _ = snake.reset()
    # print('[TRAIN: numpy state]',state.shape)
    state= torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # print('[TRAIN: tensor state]',state.shape)
    frame_stacker.push(state)
    episode_reward = 0
    steps_conter = 0
    while playing:
        steps_conter += 1
        if steps_conter % 2000 == 0:
            if len(snake.snake) > snake_length:
                snake_length = len(snake.snake)
                print('[SNAKE ATE EXTENDS EPISODE]')
                steps_conter = 0
            else:
                print('NO PROGRESS END EPISODE')
                break
        
        #print('[STATE.shape]', state.shape)  
        state_stack = frame_stacker.get_stacked_frames()
        state_stack = torch.tensor(state_stack, dtype=torch.float32).unsqueeze(0)
        # print('[state_stack.shape]', state_stack.shape)
        action = action_selector.select(state_stack)
        # print('[ACTION]', action)
        stacked_reward = 0
        for _ in range(FRAME_STACK_SIZE):
            # window.update()
            observation, reward, done, _ = snake.step(action)
            stacked_reward += reward
            # print('[REWARD]', reward)
            if done:
                break
        episode_reward += stacked_reward
        stacked_reward = torch.tensor([stacked_reward])
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        frame_stacker.push(observation)
        next_state_stack = torch.tensor(frame_stacker.get_stacked_frames()).unsqueeze(0)
        
        done = torch.tensor([done])
        memory.push(state_stack, action, next_state_stack, stacked_reward, done)
        state = observation
        frame_stacker.push(state)
        optimize_model(model=MODEL)
        if i_episode % 1000 == 0:
            hard_update_target_net()
        #update_target_net()
        _handle_user_input()

        if done.item():
            print(f"Progress: {i_episode+1}/{num_episodes}")
            rewards.append(episode_reward)
            plot(rewards)
            print("DONE!")
            torch.save(policy_net, "results/DQN.model")
            # window.update()
            break


plt.savefig("results/training_results.pdf")
plt.show()