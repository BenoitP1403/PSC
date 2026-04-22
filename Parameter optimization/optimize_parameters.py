import argparse
from study_algorithm import load_env_config, study_algorithm, Path
from objective_functions import (
    create_objective_ppo,
    create_objective_mppo,
    create_objective_dqn,
    create_objective_a2c,
)

SCENARIO_PATH = Path(__file__).parent.parent / "Scenarios" / "data_manipulation_3_pc.yaml"
N_TRIALS = 50
TOTAL_TIMESTEPS = 100_000
N_EVAL_EPISODES = 50
TUNE_SEEDS = [0, 1, 2]

OBJECTIVES = {
    "PPO":  create_objective_ppo,
    "MPPO": create_objective_mppo,
    "DQN":  create_objective_dqn,
    "A2C":  create_objective_a2c,
}

parser = argparse.ArgumentParser()
parser.add_argument("--algo", required=True, choices=OBJECTIVES.keys())
args = parser.parse_args()

study_algorithm(
    study_name=args.algo,
    create_objective=OBJECTIVES[args.algo],
    env_config=load_env_config(SCENARIO_PATH),
    n_trials=N_TRIALS,
    total_timesteps=TOTAL_TIMESTEPS,
    n_eval_episodes=N_EVAL_EPISODES,
    tune_seeds=TUNE_SEEDS,
)

"""
python optimize_parameters.py --algo PPO (or MPPO, DQN, A2C)
"""