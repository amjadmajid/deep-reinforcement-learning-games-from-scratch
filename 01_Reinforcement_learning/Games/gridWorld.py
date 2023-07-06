import numpy as np

class Action(): 
    def __init__(self):
        self.action_space = {'U': (-1, 0), 'D': (1, 0),\
                                'L': (0, -1), 'R': (0, 1)}
        self.possible_actions = list(self.action_space.keys())
        self.action_n = len(self.possible_actions)
        self.action_idxs = range(self.action_n)

    def get_action_by_idx(self, idx):
        return self.action_space[self.possible_actions[idx]]

    def get_action_by_key(self, key):
        return self.action_space[key]

class GridWorld():
    def __init__(self, shape=(3, 3), obstacles=[], terminal=None, agent_pos=(0, 0)):
        self.action  = Action()
        self.shape = shape
        self.rows = shape[0]
        self.cols = shape[1]
        self.obstacles = obstacles
        self.agent_pos = agent_pos
        self.agent_init_pos = agent_pos
        if terminal is None:
            self.terminal_state = (self.rows-1, self.cols-1)
        else:
            self.terminal_state = terminal
        self.done = False

        self.is_same_state = lambda s1, s2 : (np.array(s1)==np.array(s2)).all()

    def add_state_action(self, s, a):
        action = self.action.get_action_by_idx(a)
        return (s[0] + action[0],s[1] + action[1] )

    def is_obstacle(self, s):
        return tuple(s) in self.obstacles

    def is_terminal(self, s):
        return self.is_same_state(s, self.terminal_state)  

    def is_edge(self, s):
        if s[0] < 0 or s[0] > self.rows -1 \
            or s[1] < 0 or s[1] > self.cols -1:
            return True
        return False

    def set_agent_pos(self, state):
        self.agent_pos = state

    def get_agent_pos(self):
        return self.agent_pos
    
    def step(self, action):
        # agent location is current agent location + action
        state = self.get_agent_pos()
        tmp_state = self.add_state_action(state, action) 
        #print(f"tmp_state{tmp_state}")
        if self.is_obstacle(tmp_state):
            # print("OBSTACLES")
            pass
        elif self.is_terminal(tmp_state):
            self.set_agent_pos(tmp_state)
            #print(f"terminal_state:{tmp_state}")
            self.done = True
            #print("Done")
        elif self.is_edge(tmp_state):
            # print("Edge")
            pass
        else:        
            self.set_agent_pos(tmp_state)
        
        reward = -1 if not self.done else 0

        return self.get_agent_pos(), reward, self.done, None

    def simulated_step(self, state, action):
        if self.is_terminal(state):
            #state = next_state
            return state, 0, True, None
        # print("[ACTION]", action)
        next_state = self.add_state_action(state, action) 
        if self.is_obstacle(next_state) or self.is_edge(next_state):
            pass
        else:        
            state = next_state
        return state, -1, False, None

    def reset(self):
        self.set_agent_pos(self.agent_init_pos)
        self.done = False
        return self.agent_init_pos

    def render(self):
        for r in range(self.rows):
            for c in range(self.cols):
                state = np.array((r, c))
                if self.is_terminal(state):
                    if self.done:
                        print('[O]', end="\t")
                    else:
                        print('[]', end="\t")
                elif  self.is_same_state(self.get_agent_pos(), state): 
                    print('O', end="\t")
                elif self.is_obstacle(state):
                    print('X', end="\t")
                else:
                    print('-', end="\t")
            print()


if __name__ == '__main__':
    env = GridWorld(shape=(5, 5), obstacles=((0, 1), (1, 1)))
    env.render()
    env.step(0) # 0 -> up
    env.step(1) # 1 -> down
    env.step(2) # 2 -> left
    env.render()