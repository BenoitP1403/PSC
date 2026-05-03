import os
import numpy as np
from datetime import datetime
from config import ROWS, COLS, MAX_MOVES, ITERATIONS, LEARNING_RATE, DISCOUNT, POISON_PROB, CHEESE_SIMPLEX

SAVES_DIR = os.path.join(os.path.dirname(__file__), "saves")

LEARNING_CONFIG = {
    "ROWS": ROWS,
    "COLS": COLS,
    "MAX_MOVES": MAX_MOVES,
    "ITERATIONS": ITERATIONS,
    "LEARNING_RATE": LEARNING_RATE,
    "DISCOUNT": DISCOUNT,
    "POISON_PROB": POISON_PROB,
    "CHEESE_SIMPLEX": CHEESE_SIMPLEX,
}


def _config_summary(meta):
    return (
        f"  Grille      : {int(meta['ROWS'])}x{int(meta['COLS'])}\n"
        f"  Itérations  : {int(meta['ITERATIONS'])}\n"
        f"  Learning rate: {float(meta['LEARNING_RATE'])}\n"
        f"  Discount    : {float(meta['DISCOUNT'])}\n"
        f"  Max moves   : {int(meta['MAX_MOVES'])}\n"
        f"  Poison prob : {float(meta['POISON_PROB'])}\n"
        f"  Cheese      : {tuple(meta['CHEESE_SIMPLEX'].tolist())}"
    )


def save(algo_name, board, parameter):
    os.makedirs(SAVES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{algo_name}_{timestamp}.npz"
    path = os.path.join(SAVES_DIR, filename)
    np.savez(
        path,
        board=board,
        parameter=parameter,
        algo_name=np.array(algo_name),
        **{k: np.array(v) for k, v in LEARNING_CONFIG.items()},
    )
    print(f"Poids sauvegardés : {filename}")


def load(path):
    data = np.load(path, allow_pickle=False)
    return data["board"], data["parameter"], data


def list_saves():
    """Retourne la liste de dicts {path, filename, algo_name, meta} triés par date."""
    if not os.path.isdir(SAVES_DIR):
        return []
    saves = []
    for f in sorted(os.listdir(SAVES_DIR)):
        if not f.endswith(".npz"):
            continue
        path = os.path.join(SAVES_DIR, f)
        try:
            data = np.load(path, allow_pickle=False)
            algo = str(data["algo_name"])
            saves.append({"path": path, "filename": f, "algo_name": algo, "meta": data})
        except Exception:
            pass
    return saves


def print_save_info(filename, meta):
    print(f"  Fichier     : {filename}")
    print(_config_summary(meta))


def train_and_save(learning_fn, algo_name):
    """Wrapper : entraîne puis sauvegarde les poids."""
    def wrapper(board):
        parameter = learning_fn(board)
        save(algo_name, board, parameter)
        return parameter
    return wrapper


def load_only(save_entry):
    """Wrapper : charge les poids d'un fichier donné sans entraîner."""
    def wrapper(board):
        print(f"\nChargement de :")
        print_save_info(save_entry["filename"], save_entry["meta"])
        print()
        return save_entry["meta"]["parameter"]
    return wrapper
