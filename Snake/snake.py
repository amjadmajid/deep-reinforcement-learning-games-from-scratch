import random
from enum import Enum
from env import Vertex, Grid, Direction
from typing import List, Tuple

REWARD_FOOD = 10
REWARD_DIE = -10
REWARD_MOVE = 0


class Snake:
    def __init__(self, grid):
        self.w = grid.width
        self.h = grid.height 
        self.cz = grid.cell_size
        self.grid = grid
        self.direction = Direction(self.cz)
        self._reset()

    def _gen_food(self):
        x = random.choice(self.grid.x_vertices)
        y = random.choice(self.grid.y_vertices)
        food_v = Vertex(x,y)
        if food_v in self.snake:
            self._gen_food()
        return food_v     
    
    def move(self, dir: Vertex) -> bool:
        self.head += dir
        self.snake.insert(0, self.head)
        if not self.eat():
            self.snake.pop() 
        return self.has_died()  

    def eat(self) -> bool:
        if self.head == self.food:
            self.reward += REWARD_FOOD
            self.score +=1
            print(self.score)
            self.food = self._gen_food()
            return True
        return False 

    def has_died(self) -> bool:
        died =False
        max_x = self.grid.x_vertices[-1]
        max_y = self.grid.y_vertices[-1]
        x = self.head.x
        y = self.head.y 
        if x > max_x or y > max_y or x < 0 or y < 0 :
            self.reward += REWARD_DIE
            died  = True
        elif self._bite_itself(self.head): # check if the snake has bitten itself
            self.reward += REWARD_DIE
            died  = True
        return died  
    
    def _bite_itself(self, head: Vertex) -> bool:
        if head in self.snake[1:]:
            print("[Oach] bitten itself", )
            return True
        return False
    
    def possible_collision(self, head: Vertex) -> bool:
        # print(f"head: {head}")
        died =False
        max_x  = self.grid.x_vertices[-1]
        max_y = self.grid.y_vertices[-1]
        # print(max_x, head.x, max_y, head.y)
        if head.x >  max_x or head.y > max_y  or head.x < 0 or head.y < 0 :
            died  = True
        elif self._bite_itself(head):
            died = True
        return died  
    
    def _dir_calculator(self, action):
        # print("[Curr Dir, action]", self.dir, action)
        if action[0][0] == 1: # define it as the same dir
            pass
        if action[0][1] == 1: # define it as going right
            self.dir_idx = self.dir_idx - 1
            if self.dir_idx == -1 :
                self.dir_idx = 3 # hardcoded
        if action[0][2] == 1: # define it as going left
            self.dir_idx = self.dir_idx + 1
            if self.dir_idx == 4: 
                self.dir_idx =0
        self.dir = self.direction.directions[self.dir_idx]
        # print("[New Dir]", self.dir)
        return self.dir

    def step(self, action):
        self.reward += REWARD_MOVE
        dir = self._dir_calculator(action)
        done = self.move(dir)
        reward = self.reward
        self.reward = 0
        return self.game_state(), reward, done, None

    def game_state(self):
        # print(f" dir {self.direction.directions}: {self.dir,self.dir_idx}: {(self.dir_idx +1)%4} : {(self.dir_idx -1)%4}")
        state = [ int(self.possible_collision(self.head + self.dir)), \
                 int(self.possible_collision( self.head + self.direction.directions[(self.dir_idx +1)%4] )), 
                 int(self.possible_collision( self.head + self.direction.directions[(self.dir_idx -1)%4] ))
                 ]
        # print(f"Dnger state {state}")
        dirs_state = [0 for _ in self.direction.directions]
        dirs_state[self.dir_idx] = 1
        state = state + dirs_state
        state.append(int (self.head.x > self.food.x) )
        state.append(int (self.head.y > self.food.y))
        state.append(int (self.head.x < self.food.x) )
        state.append(int (self.head.y < self.food.y))
        return state
    
    def _reset(self):
        # becuase the way we are calculating the danger and how the snake is laid down initially 
        # we need to fix the initial direction
        self.dir_idx, self.dir = (0, self.direction.RIGHT)
        self.head =  Vertex(self.w//2, self.h//2)
        self.snake = [self.head, self.head - Vertex(self.cz, 0) , self.head - Vertex(2*self.cz, 0)]
        self.score = 0 
        self.food = self._gen_food()
        self.reward = 0
    
    def reset(self):
        self._reset()
        return self.game_state(), None
