from gridWorld import GridWorld
import time
import torch
from torch.distributions import Categorical

if __name__ == '__main__':
  
    REINFORCE = torch.load("results/REINFORCE.model")
    REINFORCE.eval()
   
    env = GridWorld(shape = (5,5), obstacles = \
                    [(0,1),  (2,1), (3,1), (2,3),(1,3),(3,3),(4,3) ])
    state, info = env.reset()
    done = False

    for i in range(100):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        print(f"state: {state}")
        action_probs = REINFORCE(state)
        action = Categorical(action_probs).sample().item()
        print(action)
        next_state, reward, done, _  = env.step(action)
        env.render()
        time.sleep(.2)
        if done:
            time.sleep(.5)
            print("DONE!")
            break
        state = next_state
    if not done: 
        print("Failed")

