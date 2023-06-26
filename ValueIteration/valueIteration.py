import numpy as np
import time 

from utils import * 
from gridWorld import GridWorld

class ValueIteration():
    def __init__(self, env, theta=1e-6, gamma=.99):
        self.env = env
        self.env_shape = env.shape
        self.actions = env.action.action_idxs
        self.actions_n = len(self.actions)
        self.policy_probs = self._init_policy_probs_table()
        self.V = {}
        self.reset_V_table()
        self.gamma = gamma
        self.theta = theta
                            
    def _init_policy_probs_table(self):
        return np.full((self.env_shape[0], self.env_shape[1], self.actions_n),\
                                1/self.actions_n)
    def _policy(self, state):
        return self.policy_probs[state[0], state[1]]
        
    def choose_action(self, state):
        print(f"Selected action: {self._policy(state)}")
        return np.random.choice(self.actions, p=self._policy(state))

    def reset_V_table(self):
        # TODO: V table must be reset when the environment is reset
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                    self.V[(r, c)] = 0

    def value_iteration(self):
        delta = float("inf")
        
        while delta > self.theta: 
            delta = 0 

            for state in self.V:
                # time.sleep(.1)
                if self.env.is_terminal(state):
                    break
                old_value = self.V[state]
                # keep track of the action probabilities 
                action_probs = None
                # keep track of the max Q value
                max_qsa = float('-inf')

                # look for the max Q value back taking fake (simulated) steps
                for action in self.actions:
                    next_state, r, _,_ = self.env.simulated_step(state, action)
                    qsa = r + self.gamma * self.V[(next_state[0], next_state[1])]

                    # identify the action with max Q value
                    if qsa > max_qsa:
                        max_qsa = qsa
                        action_probs = np.zeros(4)
                        action_probs[action] = 1.0
                    
                # update value state table 
                self.V[state] = max_qsa
                # update the probabilities of choosing an action
                # this is a deterministic policy
                self.policy_probs[state] = action_probs
                
                # keep the maximum difference of a state value
                delta = max(delta, abs(max_qsa - old_value))


if __name__ == "__main__":
    env = GridWorld(shape = (5,5), obstacles = [(0,1), (1,1), (2,1), (4,1),\
                    (0,3),(2,3),(3,3), (4,3) ])
    
    state = env.reset()
    agent = ValueIteration(env)
    agent.value_iteration()
    steps =0

    # Interact with the environment based on the policy derived from value iteration
    while True: 
        steps += 1
        clear_screen()
        action = agent.choose_action(state)
        state, _, done, _ = env.step(action)
        env.render()
        if done:
            print(f"the agent reached terminal state in {steps} steps")
            plot_state_value_table(agent.V, env.cols)
            break

        time.sleep(.5)