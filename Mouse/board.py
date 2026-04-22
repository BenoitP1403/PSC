from random import random, choices
import numpy as np


def get_rand_pos(rows, cols, forbidden):
    while True:
        rand_pos = int(random() * cols), int(random() * rows)
        if rand_pos not in forbidden:
            return rand_pos


def generate_board(rows, cols, poison_prob, cheese_simplex):
    board = np.zeros((rows, cols), dtype=np.int8)

    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                continue

            if random() < poison_prob:
                board[r, c] = -1
            else:
                board[r, c] = choices(range(len(cheese_simplex)), cheese_simplex, k=1)[0]

    poison_x, poison_y = get_rand_pos(rows, cols, [(0, 0)])
    board[poison_y, poison_x] = -1
    cheese_x, cheese_y = get_rand_pos(rows, cols, [(0, 0), (poison_x, poison_y)])
    board[cheese_y, cheese_x] = choices(range(1, len(cheese_simplex)), cheese_simplex[1:], k=1)[0]

    return board