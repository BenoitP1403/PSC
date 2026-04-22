import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import optuna
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from primaite.session.environment import PrimaiteGymEnv


logger = logging.getLogger("primaite")
logger.handlers.clear()
logger.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BEST_PARAMS_DIR = Path(__file__).parent / "best_params"
BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def load_env_config(scenario_path: str) -> Dict[str, Any]:
    with open(scenario_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_env(env_config: Dict[str, Any]) -> PrimaiteGymEnv:
    return PrimaiteGymEnv(env_config=copy.deepcopy(env_config))


def create_study(
    study_name: str,
    direction: str = "maximize",
    storage_path: Path = BEST_PARAMS_DIR / "optuna_studies.db",
    n_startup_trials: int = 10,
    n_warmup_steps: int = 1,
) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        ),
    )


def save_best_result(
    output_path: Union[str, Path],
    study: optuna.Study,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "trial_user_attrs": best.user_attrs,
        "n_trials_completed": len(study.trials),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload["metadata"] = metadata

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def study_algorithm(
    study_name: str,
    create_objective: function,
    env_config: Dict[str, Any],
    n_trials: int,
    total_timesteps: int,
    n_eval_episodes: int,
    tune_seeds: List[int],
    output_path: Optional[Union[str, Path]] = None,
) -> optuna.Study:
    if output_path is None:
        output_path = BEST_PARAMS_DIR / f"{study_name}.json"

    objective = create_objective(
        env_config=env_config,
        total_timesteps=total_timesteps,
        n_eval_episodes=n_eval_episodes,
        tune_seeds=tune_seeds,
    )
    metadata = {
        "algorithm":       study_name,
        "total_timesteps": total_timesteps,
        "eval_episodes":   n_eval_episodes,
        "tune_seeds":      tune_seeds,
    }

    study = create_study(study_name=study_name)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    save_best_result(output_path=output_path, study=study, metadata=metadata)
    return study