import pygame
import sys
from board import *
from env import *
from config import *

TILE_SIZE = 30
WIDTH = COLS * TILE_SIZE
HEIGHT = ROWS * TILE_SIZE
FPS = 30

COLOR_BG = (20, 20, 20)
COLOR_GRID = (40, 40, 40)
COLOR_EMPTY = (120, 120, 120)
COLOR_POISON = (220, 50, 50)
COLOR_CHEESE = (250, 200, 0)
COLOR_MOUSE = (255, 255, 255)
COLOR_GAMEOVER = (255, 0, 0)


def draw_board(screen, rows, cols, board, font):
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            value = board[r, c]

            color = COLOR_EMPTY
            text = ""

            if value == -1:
                color = COLOR_POISON
                text = "P"
            elif value > 0:
                color = COLOR_CHEESE
                text = str(value)

            pygame.draw.rect(screen, color, rect)

            if text:
                text_surf = font.render(text, True, COLOR_BG)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)

            pygame.draw.rect(screen, COLOR_GRID, rect, 1)


def draw_mouse(screen, state):
    x, y = state
    center_x = int((x + 0.5) * TILE_SIZE)
    center_y = int((y + 0.5) * TILE_SIZE)
    radius = int(TILE_SIZE * 0.35)

    pygame.draw.circle(screen, COLOR_MOUSE, (center_x, center_y), radius)


def draw_game_over(screen, message, font):
    text_surf = font.render(message, True, COLOR_GAMEOVER)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))

    bg_rect = pygame.Rect(0, 0, text_rect.width + 40, text_rect.height + 40)
    bg_rect.center = text_rect.center
    bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
    bg_surf.fill((0, 0, 0, 180))

    screen.blit(bg_surf, bg_rect)
    screen.blit(text_surf, text_rect)


def visualize(learning_function, policy):
    print("Generating board...")
    BOARD = generate_board(ROWS, COLS, POISON_PROB, CHEESE_SIMPLEX)
    print("Started training...")
    PARAMETER = learning_function(BOARD)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    tile_font = pygame.font.SysFont('Arial', 24, bold=True)
    game_over_font = pygame.font.SysFont('Arial', 12, bold=True)

    viz_board = BOARD.copy()
    state = (0, 0)
    moves = 0
    game_over = False
    game_over_message = ""

    running = True
    print("Press any key to advance one step.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if not game_over:
                    action = policy(state, PARAMETER)

                    if action is None:
                        game_over = True
                        game_over_message = "Agent stuck or no valid moves!"
                        continue

                    new_state = get_new_state(state, action)
                    r, c = new_state[1], new_state[0]

                    if viz_board[r, c] > 0:
                        viz_board[r, c] -= 1

                    state = new_state
                    moves += 1

                    terminal, reason = is_terminal(state, viz_board)
                    if terminal:
                        game_over = True
                        if reason == "poison":
                            game_over_message = "GAME OVER: ATE POISON"
                        elif reason == "ate all cheese":
                            game_over_message = "SUCCESS: ATE ALL CHEESE!"
                    elif moves >= MAX_MOVES:
                        game_over = True
                        game_over_message = "GAME OVER: OUT OF MOVES"

        screen.fill(COLOR_BG)
        draw_board(screen, ROWS, COLS, viz_board, tile_font)
        draw_mouse(screen, state)

        if game_over:
            draw_game_over(screen, game_over_message, game_over_font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
