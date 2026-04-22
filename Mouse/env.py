import numpy as np


def get_new_state(state, action):
    return state[0] + action[0], state[1] + action[1]


def is_terminal(state, board):
    if board[state[1], state[0]] == -1:
        return True, "poison"
    if np.sum(board[board > 0]) == 0:
        return True, "ate all cheese"
    return False, "running"


def is_valid_action(state, action, rows, cols):
    new_x, new_y = get_new_state(state, action)
    return 0 <= new_x < cols and 0 <= new_y < rows


def get_valid_actions(state, actions, rows, cols):
    return [action for action in actions if is_valid_action(state, action, rows, cols)]


def find_nearest_cheese_distance(state, board):
    x, y = state
    rows, cols = board.shape
    min_distance = float('inf')

    for r in range(rows):
        for c in range(cols):
            if board[r, c] > 0:
                distance = abs(x - c) + abs(y - r)
                min_distance = min(min_distance, distance)

    return min_distance if min_distance != float('inf') else rows + cols


def get_reward(state, board):
    value = board[state[1], state[0]]

    if value == -1:
        return -1000

    if value > 0:
        return 50

    distance = find_nearest_cheese_distance(state, board)
    proximity_bonus = max(0, 5 - distance)
    return -5 + proximity_bonus
