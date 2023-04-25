from gridWorld import GridWorld
import numpy as np
import time
import torch

if __name__ == '__main__':
    dqn = torch.load("policy_net")
    dqn.eval()
    # for param in dqn.parameters():
    #     print(param)
    env = GridWorld(shape = (5,5), obstacles = \
                    [(0,1), (1,1), (2,1), (3,1), (0,3),(2,3),(3,3),(1,3) ])
    state, info = env.reset()
    done = False

    for i in range(100):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = dqn(state).max(1)[1].view(1, 1)
        print(action)
        next_state, reward, done, _  = env.step(action.item())
        env.render()
        time.sleep(.2)
        if done:
            time.sleep(.5)
            print("DONE!")
            break
        state = next_state
    if not done: 
        print("Failed")

