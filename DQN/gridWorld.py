import numpy as np 
import time
import os 

class GridWorld():
    def __init__(self, 
                 shape=(3, 3), 
                 obstacles=[], 
                 terminal=None, 
                 agent_pos=(0,0),
                  #               up     down   left     right
                 action_space=[(-1, 0), (1, 0),(0, -1), (0, 1)]
                 ):
        self.agent_init_pos = agent_pos
        self.agent_pos = agent_pos
        self.action_space = action_space
        self.rows = shape[0]
        self.cols = shape[1]
        self.obstacles = obstacles
        if terminal == None:
            self.terminal_state = (self.rows-1, self.cols-1)
        else:
            self.terminal_state = terminal
        self.done = False

        self.add_state_action = lambda s, a : tuple(np.array(s) + np.array(a))
        self.is_same_state = lambda s1, s2 : (np.array(s1)==np.array(s2)).all()

        # clear the terminal (or command line)
        # TODO: this function run on linux only. 
        # for windows the clear must be replaced with cls
        self.clear = lambda: os.system('clear')

    def is_obstacle(self, pos):
        if pos in self.obstacles:
            return True
        return False

    def is_terminal(self, s):
        return self.is_same_state(s, self.terminal_state)

    def is_outside(self, pos):
        if pos[0] < 0 or pos[0] > (self.rows -1) \
            or pos[1] < 0 or pos[1] > (self.cols -1):
            return True
        return False

    def set_pos(self, pos):
        # check if agent is not None
        self.agent_pos = pos

    def get_pos(self):
        return self.agent_pos

    def _next_pos(self, action):
        next_state = self.add_state_action(self.get_pos(), action) 
        if self.is_obstacle(next_state) or self.is_outside(next_state):
            return 0
        return 1

    def game_state(self):
      state = [ self._next_pos(action) for action in self.action_space]
      state.append(self.get_pos()[0])
      state.append(self.get_pos()[1])
      return state
        
    def step(self, action_idx ):
        agent_pos = self.get_pos()
        action = self.action_space[action_idx]
        tmp_pos = self.add_state_action(agent_pos, action) 
        if self.is_obstacle(tmp_pos):
            pass
        elif self.is_terminal(tmp_pos):
            self.set_pos(tmp_pos)
            self.done = True
        elif self.is_outside(tmp_pos):
            pass
        else:        
            self.set_pos(tmp_pos)
        
        reward = -1 if not self.done else 0

        return self.game_state(), reward, self.done, None

    def reset(self):
        self.set_pos(self.agent_init_pos)
        self.done = False
        return self.game_state(), None

    def render(self):
        self.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if self.is_terminal(state):
                    if self.done:
                        print('[O]', end="\t")
                    else:
                        print('[]', end="\t")
                elif  self.is_same_state(self.get_pos(), state): 
                    print('O', end="\t")
                elif self.is_obstacle(state):
                    print('X', end="\t")
                else:
                    print('-', end="\t")
            print()


if __name__ == "__main__":
    env = GridWorld(shape = (5,5), 
                    agent_pos=(0,0),
                    terminal=None,
                    obstacles = [(0,1),(1,1), (2,1), (3,1),(2,3),(3,3),(4,3) ])
    state, _ = env.reset()
    env.render()
    state, _ , __, ___ = env.step(1)
    time.sleep(.5)
    env.render()    

    state, _ , __, ___ = env.step(1)
    time.sleep(.5)
    env.render()    

    state, _ , __, ___ = env.step(1)
    time.sleep(.5)
    env.render()    
    
    state, _ , __, ___ = env.step(1)
    time.sleep(.5)
    env.render()    

    state, _ , __, ___ = env.step(3)
    print(state)
    time.sleep(.5)
    env.render()    

