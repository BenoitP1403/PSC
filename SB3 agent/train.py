import logging
import argparse
import yaml
import json
from pathlib import Path
from primaite.session.environment import PrimaiteGymEnv
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import MaskablePPO

logger = logging.getLogger("primaite")
logger.handlers.clear()
logger.propagate = False

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Paths resolved relative to this file, so the script works from any cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PARAMS_DIR = REPO_ROOT / "Parameter optimization" / "best_params"
SCENARIO_PATH = REPO_ROOT / "Scenarios" / "data_manipulation_3_pc.yaml"


with open(SCENARIO_PATH, "r") as file:
    cfg = yaml.safe_load(file)

cfg["io_settings"]["save_agent_actions"] = False
env = PrimaiteGymEnv(env_config=cfg)


def load_params(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["best_params"]


TOTAL_TIMESTEPS = 1_000_000


ALGORITHMS = {
    "PPO": (PPO, load_params(PARAMS_DIR / "PPO.json")),
    "MPPO": (MaskablePPO, load_params(PARAMS_DIR / "MPPO.json")),
    "A2C": (A2C, load_params(PARAMS_DIR / "A2C.json")),
    "DQN": (DQN, load_params(PARAMS_DIR / "DQN.json")),
}

parser = argparse.ArgumentParser()
parser.add_argument("--algo", required=True, choices=ALGORITHMS.keys())
args = parser.parse_args()

algorithm, params = ALGORITHMS[args.algo]
model = algorithm(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=str(SCRIPT_DIR / "tensorboard_logs" / args.algo),
    **params,
)
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(str(SCRIPT_DIR / f"{args.algo} model.zip"))

print("Training finished.")

"""
python train.py --algo PPO (or MPPO, A2C, DQN)
"""
