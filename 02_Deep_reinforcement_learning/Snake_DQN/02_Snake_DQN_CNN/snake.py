import random
from env import Vertex, Direction
from typing import List, Tuple

# Define reward constants
REWARD_FOOD = 10000
REWARD_DIE = -1000
REWARD_MOVE = -1

class Snake:
    """
    The Snake class implements the behavior of the snake in the game. 
    It holds the position and size of the snake, and handles the logic of moving, eating food and dying.
    """
    def __init__(self, grid):
        self.w = grid.width
        self.h = grid.height 
        self.cz = grid.cell_size
        self.grid = grid
        self.direction = Direction(self.cz)
        self._reset()

    def _gen_food(self):
        """
        Generate a new food item at a random position in the grid.
        """
        x = random.choice(self.grid.x_vertices[1:-1]) # Don't generate food on the terminal states
        y = random.choice(self.grid.y_vertices[1:-1]) 
        food_v = Vertex(x,y)
        if food_v in self.snake:  # If food generated inside the snake, generate again
            return self._gen_food()
        return food_v     

    def move(self, dir: Vertex) -> bool:
        """
        Move the snake in the specified direction. Return if the snake has died as a result of the move.
        """
        self.head += dir
        self.snake.insert(0, self.head)
        if not self.eat():  # If no food was eaten, remove the last segment of the snake
            self.snake.pop() 
        return self.has_died()  

    def eat(self) -> bool:
        """
        Check if the snake's head is on a food item. If so, increase score and generate new food.
        """
        if self.head == self.food:
            self.reward += REWARD_FOOD
            self.score += 1
            self.food = self._gen_food()
            return True
        return False 

    def has_died(self) -> bool:
        """
        Check if the snake has died either by moving out of the grid or by running into itself.
        """
        died = False
        max_x = self.grid.x_vertices[-2] # if you go to the last vertex, you are out of bounds
        max_y = self.grid.y_vertices[-2]
        x = self.head.x
        y = self.head.y 
        # Check if the snake is out of bounds
        # print("[Oach] x:", x, "y:", y, "max_x:", max_x, "max_y:", max_y)
        if x > max_x or y > max_y or x < 0 or y < 0 :
            self.reward += REWARD_DIE
            died = True
        # Check if the snake has bitten itself
        elif self._bite_itself(self.head): 
            self.reward += REWARD_DIE
            died = True
        return died  

    def _bite_itself(self, head: Vertex) -> bool:
        """
        Check if the snake has bitten itself.
        """
        if head in self.snake[1:]:
            # print("[Oach] bitten itself")
            return True
        return False
    
    def possible_collision(self, head: Vertex) -> bool:
        """
        Check if moving to a new position would cause the snake to die.
        """
        died = False
        max_x  = self.grid.x_vertices[-2]
        max_y = self.grid.y_vertices[-2]
        # Check if the new position is out of bounds
        if head.x >  max_x or head.y > max_y  or head.x < 0 or head.y < 0 :
            died  = True
        # Check if the snake would bite itself
        elif self._bite_itself(head):
            died = True
        return died  

    def _dir_calculator(self, action):
        """
        Update the direction based on the action.
        """
        if action[0][0] == 1: # define it as the same direction
            pass
        if action[0][1] == 1: # define it as turning right
            self.dir_idx = self.dir_idx - 1
            if self.dir_idx == -1 :
                self.dir_idx = 3
        if action[0][2] == 1: # define it as turning left
            self.dir_idx = self.dir_idx + 1
            if self.dir_idx == 4: 
                self.dir_idx = 0
        self.dir = self.direction.directions[self.dir_idx]
        return self.dir

    def step(self, action):
        """
        Perform one step in the game. Move the snake according to the action, update the grid, and return the new state and reward.
        """
        self.reward = REWARD_MOVE
        dir = self._dir_calculator(action)
        done = self.move(dir)
        self.grid.update(self)
        return self.game_state(), self.reward, done, None

    def game_state(self):
        """
        Return the current state of the game.
        """
        return self.grid.grid

    def _reset(self):
        """
        Reset the snake to its initial state.
        """
        self.dir_idx, self.dir = (0, self.direction.RIGHT)
        self.head =  Vertex(self.w//2, self.h//2)
        self.snake = [self.head, self.head - Vertex(1, 0) , self.head - Vertex(2, 0)]
        self.score = 0 
        self.food = self._gen_food()
        self.reward = 0
        self.grid.update(self) 
    
    def reset(self):
        """
        Reset the game and return the initial state.
        """
        self._reset()
        return self.game_state(), None

