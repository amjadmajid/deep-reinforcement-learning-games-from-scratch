from gridWorld import GridWorld
import time
import torch

if __name__ == '__main__':
    dqn = torch.load("results/DQN.model")
    dqn.eval()
   
    env = GridWorld(shape = (5,5), obstacles = \
                     [(0,1), (1,1), (4,1), (3,1),(2,3),(3,3),(4,3) ])
    state, info = env.reset()
    done = False

    for i in range(100):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = dqn(state).max(1)[1].view(1, 1).item()
        next_state, reward, done, _  = env.step(action)
        state = next_state
        env.render()
        print(f"{action=}")
        time.sleep(.2)
        if done:
            time.sleep(.5)
            print("DONE!")
            break
    if not done: 
        print("Failed")

