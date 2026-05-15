import pygame
import numpy as np
from enum import Enum
from collections import namedtuple
import random

# --- Constants ---
BLOCK_SIZE = 20
SPEED = 40  # FPS (higher = faster game, faster training)
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)

class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

Point = namedtuple('Point', 'x, y')


class SnakeGame:
    """
    Core Snake game logic.
    This class handles the game state — it doesn't know about RL yet.
    The Gymnasium wrapper (next step) will sit on top of this.
    """

    def __init__(self, width=480, height=480, render=True):
        self.width  = width
        self.height = height
        self.render_mode = render

        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('RL Snake Bot')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 25)

        self.reset()

    def reset(self):
        """Reset game to initial state. Called at start of every episode."""
        # Snake starts in the middle, moving right
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food  = None
        self._place_food()
        self.frame_iteration = 0  # track steps (to detect infinite loops)

    def _place_food(self):
        """Place food at a random position not occupied by the snake."""
        x = random.randrange(0, self.width, BLOCK_SIZE)
        y = random.randrange(0, self.height, BLOCK_SIZE)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()  # try again if food lands on snake

    def play_step(self, action):
        """
        Execute one game step.
        action: [1,0,0] = straight, [0,1,0] = right turn, [0,0,1] = left turn

        Returns: reward, game_over (bool), score
        """
        self.frame_iteration += 1

        # 1. Handle quit event (so the window doesn't freeze)
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. Move snake
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False

        # Die if: hit wall, hit self, OR stuck in loop (no food found)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # remove tail (snake moves forward)
            reward = 0        # small reward for staying alive

        # 5. Update display
        if self.render_mode:
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, point=None):
        """Check if a point (default: head) hits a wall or the snake body."""
        if point is None:
            point = self.head
        # Wall collision
        if (point.x > self.width - BLOCK_SIZE or point.x < 0 or
                point.y > self.height - BLOCK_SIZE or point.y < 0):
            return True
        # Self collision
        if point in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        """
        Convert action to direction.
        Actions are RELATIVE: [straight, right_turn, left_turn]
        This is better than absolute (UP/DOWN/LEFT/RIGHT) because
        the agent only needs 3 outputs instead of 4.
        """
        # Clock-wise order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]               # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]     # right turn
        else:
            new_dir = clock_wise[(idx - 1) % 4]     # left turn

        self.direction = new_dir

        x, y = self.head
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP:   y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _update_ui(self):
        """Draw everything on screen."""
        self.display.fill(BLACK)

        # Draw snake
        for i, pt in enumerate(self.snake):
            color = BLUE1 if i == 0 else BLUE2  # head is brighter
            pygame.draw.rect(self.display, color,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE,
                             pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()

    def get_state(self):
        """
        Build the state vector the RL agent will observe.
        11 values total (all binary 0/1 except distances):

        Danger signals (3):
          - danger straight ahead
          - danger to the right
          - danger to the left

        Current direction (4, one-hot):
          - moving left, right, up, down

        Food location relative to head (4):
          - food is left, right, above, below
        """
        head = self.head
        # Points one step ahead in each direction
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y,  # food down
        ]

        return np.array(state, dtype=int)