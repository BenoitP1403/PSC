from random import choice
from env import *
from board import *
from config import *

import numpy as np


def init_Q_table(rows, cols, num_actions):
    return np.zeros((rows, cols, num_actions), dtype=np.float32)


def get_max_reachable_value(state, Q_table):
    r, c = state[1], state[0]
    valid_action_indices = [ACTION_TO_INDEX[a] for a in ACTIONS if is_valid_action(state, a, ROWS, COLS)]
    if not valid_action_indices:
        return 0
    return np.max(Q_table[r, c, valid_action_indices])


def eps_greedy_policy(state, Q_table, eps):
    valid_actions = get_valid_actions(state, ACTIONS, ROWS, COLS)
    if not valid_actions:
        return None

    if random() < eps:
        return choice(valid_actions)

    r, c = state[1], state[0]
    valid_q_values = []
    for action in valid_actions:
        idx = ACTION_TO_INDEX[action]
        valid_q_values.append((Q_table[r, c, idx], action))

    max_q = max(q for q, a in valid_q_values)
    best_actions = [a for q, a in valid_q_values if q == max_q]
    return choice(best_actions)


def greedy_policy(state, Q_table):
    return eps_greedy_policy(state, Q_table, 0)


def learn_q_table(board):
    Q_table = init_Q_table(ROWS, COLS, NUM_ACTIONS)
    eps = 1
    eps_decay = 0.999

    for i in range(ITERATIONS):
        board_ = board.copy()
        moves = 0
        state = (0, 0)

        terminal, _ = is_terminal(state, board_)

        while not terminal and moves < MAX_MOVES:
            action = eps_greedy_policy(state, Q_table, eps)
            if action is None:
                break

            new_state = get_new_state(state, action)
            reward = get_reward(new_state, board_)
            if board_[new_state[1], new_state[0]] > 0:
                board_[new_state[1], new_state[0]] -= 1

            terminal, _ = is_terminal(new_state, board_)
            action_idx = ACTION_TO_INDEX[action]
            r, c = state[1], state[0]

            old_value = Q_table[r, c, action_idx]

            if terminal:
                max_future_value = 0.0
            else:
                max_future_value = get_max_reachable_value(new_state, Q_table)

            Q_table[r, c, action_idx] = old_value + LEARNING_RATE * (reward + DISCOUNT * max_future_value - old_value)
            state = new_state
            moves += 1

        if eps > 0.01:
            eps *= eps_decay

        if i % 10000 == 0:
            print(f"\rTraining progress: {i / ITERATIONS * 100:.1f}%", end='')

    print("\nTraining finished.")
    return Q_table
