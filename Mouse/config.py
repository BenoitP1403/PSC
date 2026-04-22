ROWS = 10
COLS = 10
MAX_MOVES = 50
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # 0:Left, 1:Right, 2:Up, 3:Down
POISON_PROB = .1
CHEESE_SIMPLEX = (50, 10)
ITERATIONS = 500_000
LEARNING_RATE = 0.0005
DISCOUNT = 0.99

ACTION_TO_INDEX = {
    (-1, 0): 0,
    (1, 0): 1,
    (0, -1): 2,
    (0, 1): 3
}
NUM_ACTIONS = len(ACTIONS)