import pygame 
import random

pygame.init()
# set the font for pygame
font = pygame.font.SysFont('arial', 24)
pygame.display.set_caption("Snake")

# define color constants
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0,0,0)
GRAY = (30,30,30)
RED = (255, 0, 0)

SPEED = 40
GAP = 1
CELL = 16 # all moves will be CELL based 

class Grid:
    def __init__(self, w=640, h=480, cell=CELL):
        self.cell_size = cell
        self.width = (w // cell) * cell  # make the width multiple of the cell size
        self.height = (h // cell) * cell
        self.x_vertices = range(0, self.width, self.cell_size) 
        self.y_vertices = range(0, self.height, self.cell_size)

class Vertex:
    '''a vertex represents the top left corner of a square'''
    def __init__(self, x, y ):
        self.x = x 
        self.y = y  
        # print(s)
    def __add__(self, p):
        return Vertex( self.x  + p.x , self.y + p.y)
    def __sub__(self, p):
        return Vertex( self.x  - p.x, self.y - p.y) 
    def __eq__(self, p):
        if not isinstance(p, Vertex):
            # don't compare againt unrelated types
            return NotImplemented
        return self.x == p.x and self.y == p.y
    def __repr__(self):
        return f'Vertex{self.x, self.y}'

class Direction():
    def __init__(self, cell_size):
        self.RIGHT = Vertex(cell_size, 0)
        self.LEFT  = Vertex(-cell_size, 0)
        self.DOWN  = Vertex(0, cell_size)
        self.UP    = Vertex(0, -cell_size)
        self.directions = [self.RIGHT, self.DOWN, self.LEFT, self.UP]

    def random_dir(self) -> Vertex:
        idx = random.randint(0, len(self.directions)-1)
        return idx, self.directions[idx]

class Window:
    def __init__(self, grid, snake):
        self.snake = snake   
        self.grid = grid
        self.cz = grid.cell_size    
        self.w = grid.width
        self.h = grid.height
        self.display = pygame.display.set_mode((self.w, self.h ))
        self.clock = pygame.time.Clock()

    def draw_grid(self, vertex=False):
        for i in self.grid.x_vertices:
            for j in self.grid.y_vertices:
                rect = (i, j, self.cz-1, self.cz-1)
                pygame.draw.rect(self.display, (BLACK), rect)
                if vertex:
                    vertex_point = (i, j, 2, 2)
                    pygame.draw.rect(self.display, RED, vertex_point)

    def update(self):
        self.display.fill(GRAY)
        self.draw_grid(True)
        for pt in self.snake.snake:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x, pt.y, self.cz-GAP, self.cz-GAP))
        text = font.render(f"Score: {self.snake.score}", True, WHITE)
        self.display.blit(text, [0,0])
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.snake.food.x, self.snake.food.y, self.cz, self.cz))
        self.clock.tick(SPEED)
        pygame.display.flip()