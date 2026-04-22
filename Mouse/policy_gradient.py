import numpy as np
from env import *
from board import *
from config import *


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def policy(state, theta):
    r, c = state[1], state[0]
    preferences = theta[r, c, :].copy()
    
    for i, action in enumerate(ACTIONS):
        if not is_valid_action(state, action, ROWS, COLS):
            preferences[i] = -1e9
            
    probs = softmax(preferences)
    action_idx = np.random.choice(NUM_ACTIONS, p=probs)
    return ACTIONS[action_idx]


def learn_reinforce(board):
    rows, cols = board.shape
    theta = np.zeros((rows, cols, NUM_ACTIONS), dtype=np.float32)
    
    for i in range(ITERATIONS):
        board_ = board.copy()
        state = (0, 0)
        trajectory = []
        
        for _ in range(MAX_MOVES):
            action = policy(state, theta)
            new_state = get_new_state(state, action)
            reward = get_reward(new_state, board_)
            if board_[new_state[1], new_state[0]] > 0:
                board_[new_state[1], new_state[0]] -= 1
            
            trajectory.append((state, ACTION_TO_INDEX[action], reward))
            
            state = new_state
            terminal, _ = is_terminal(state, board_)
            if terminal: break

        G = 0
        for t in reversed(range(len(trajectory))):
            s_t, a_idx, r_t = trajectory[t]
            G = r_t + DISCOUNT * G
            
            r, c = s_t[1], s_t[0]
            probs = softmax(theta[r, c, :])
            
            for action_i in range(NUM_ACTIONS):
                if action_i == a_idx:
                    theta[r, c, action_i] += LEARNING_RATE * G * (1 - probs[action_i])
                else:
                    theta[r, c, action_i] += LEARNING_RATE * G * (-probs[action_i])

        if i % 1000 == 0:
            print(f"\rTraining progress: {i / ITERATIONS * 100:.1f}%", end='')
            
    print("\nTraining finished.")
    return theta
