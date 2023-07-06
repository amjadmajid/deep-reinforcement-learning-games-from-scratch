import pygame 
import random
import numpy as np

# Initialize pygame
pygame.init()

# Set the font for pygame
font = pygame.font.SysFont('arial', 24)
pygame.display.set_caption("Snake Game")

# Define color constants
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
GRAY = (30, 30, 30)
RED = (255, 0, 0)

# Define game speed and gap constants
SPEED = 40
GAP = 1

class Grid:
    """
    Grid class for maintaining the game grid and its state
    """
    def __init__(self, grid_size=(30,40), cell_size=16):
        self.cell_size = cell_size
        # self.width and self.height are used to define the size of the game window 
        # without the terminal states
        self.width = grid_size[0] # Ensure width is a multiple of the cell size
        self.height = grid_size[1]  # Ensure height is a multiple of the cell size
        # I added to terminal states so that when the game terminates I return a 
        # state (full grid) and not None
        self.x_vertices = range(-1, self.width + 1)  # include the termination states
        self.y_vertices = range(-1, self.height + 1)
        self.grid = np.zeros((grid_size[0]+2, grid_size[1]+2), dtype=int)  # Grid state
        self._terminal_states_labeling()

    def _terminal_states_labeling(self):
        self.grid[0, :] = -1
        self.grid[-1, :] = -1
        self.grid[:, -1] = -1
        self.grid[:, 0] = -1
    
    def update(self, snake):
        """
        Update the grid state with the current snake and food positions
        """
        self.grid.fill(0)
        self._terminal_states_labeling()
        for cell in snake.snake:
            # print('[CELL]', cell.x, cell.y)
            self.grid[cell.x, cell.y] = -1
        self.grid[snake.food.x, snake.food.y] = 1

        return self.grid

class Vertex:
    """
    Vertex class for representing the top left corner of a square
    """

    def __init__(self, x, y ):
        self.x = x 
        self.y = y  

    def __add__(self, p):
        """
        Define addition of two vertices
        """
        return Vertex( self.x  + p.x , self.y + p.y)

    def __sub__(self, p):
        """
        Define subtraction of two vertices
        """
        return Vertex( self.x  - p.x, self.y - p.y) 

    def __eq__(self, p):
        """
        Define equality of two vertices
        """
        if not isinstance(p, Vertex):
            return NotImplemented
        return self.x == p.x and self.y == p.y

    def __repr__(self):
        """
        Define string representation of a vertex
        """
        return f'Vertex{self.x, self.y}'

class Direction:
    """
    Direction class for defining possible directions for the snake to move
    """

    def __init__(self, cell_size):
        self.RIGHT = Vertex(1, 0)
        self.LEFT  = Vertex(-1, 0)
        self.DOWN  = Vertex(0, 1)
        self.UP    = Vertex(0, -1)
        self.directions = [self.RIGHT, self.DOWN, self.LEFT, self.UP]

    def random_dir(self) -> Vertex:
        """
        Returns a random direction from the defined directions
        """
        idx = random.randint(0, len(self.directions)-1)
        return idx, self.directions[idx]

class Window:
    """
    Window class for displaying the game grid and handling its visual updates
    """

    def __init__(self, grid, snake):
        self.snake = snake   
        self.grid = grid
        self.cz = grid.cell_size    
        self.w = grid.width
        self.h = grid.height
        # The set_mode function parameters causes the terminal states to be visually hidden
        self.display = pygame.display.set_mode(( int(self.w * self.cz), int(self.h * self.cz)))
        self.clock = pygame.time.Clock()

    def draw_grid(self, vertex=False):
        """
        Draw the game grid on the display window
        """
        for i in self.grid.x_vertices:
            for j in self.grid.y_vertices:
                rect = (i * self.cz, j * self.cz, self.cz-1, self.cz-1)
                color = RED if i == -1 or i == self.grid.x_vertices[-1] \
                or j == -1 or j ==self.grid.y_vertices[-1] else BLACK
                pygame.draw.rect(self.display, color, rect)
                if vertex:
                    vertex_point = (i * self.cz, j * self.cz, 2, 2)
                    pygame.draw.rect(self.display, RED, vertex_point)

    
    def _vertix_to_cell(self, vertex):
        """
        Convert a vertex to a cell
        """
        return Vertex(vertex.x * self.cz, vertex.y * self.cz)
    
    def update(self):
        """
        Update the game window with the current state of the grid
        """
        self.display.fill(GRAY)
        self.draw_grid(True)
        for pt in self.snake.snake:
            pt = self._vertix_to_cell(pt)
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x, pt.y, self.cz-GAP, self.cz-GAP))
        text = font.render(f"Score: {self.snake.score}", True, WHITE)
        self.display.blit(text, [0,0])
        food = self._vertix_to_cell(self.snake.food)
        pygame.draw.rect(self.display, GREEN, pygame.Rect(food.x, food.y, self.cz, self.cz))
        self.clock.tick(SPEED)
        pygame.display.flip()


