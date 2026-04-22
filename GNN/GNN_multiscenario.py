from __future__ import annotations
import copy
import logging
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
import matplotlib
import numpy as np
import torch
import torch.optim as optim

matplotlib.use("Agg")

from GNN_single_scenario import (
    LOG,
    TrainingConfig,
    build_agent_from_schema,
    build_universal_schema,
    collect_rollout,
    format_metric,
    linear_schedule,
    load_cfg,
    load_checkpoint,
    make_graph_env_from_cfg,
    ppo_update,
    resolve_device,
    resolve_yaml_path,
    save_checkpoint,
    set_optimizer_lr,
    set_seed,
)

SCENARIO_DIR = Path("../Scenarios")

SCENARIO_FILE_ORDER = [
    "data_manipulation.yaml",
    "data_manipulation_with_different_connections.yaml",
    "data_manipulation_plus_one_client.yaml",
    "data_manipulation_plus_one_dynamic_client.yaml",
    "data_manipulation_plus_two_dynamic_clients.yaml",
]

SCENARIO_PATHS_BY_NAME = {name: SCENARIO_DIR / name for name in SCENARIO_FILE_ORDER}

DEFAULT_CHECKPOINT_DIR = Path("checkpoints_GNN_data_manipulation_multiscenario")
DEFAULT_PLOT_DIR = Path("training_plots_GNN_data_manipulation_multiscenario")
DEFAULT_SEED = 7


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start_update: int
    end_update: Optional[int]
    scenario_names: list[str]
    sampling_weights: dict[str, float]


CURRICULUM_STAGES = [
    CurriculumStage(
        name="stage_0",
        start_update=0,
        end_update=399,
        scenario_names=[
            "data_manipulation.yaml",
        ],
        sampling_weights={
            "data_manipulation.yaml": 1.0,
        },
    ),
    CurriculumStage(
        name="stage_1",
        start_update=400,
        end_update=799,
        scenario_names=[
            "data_manipulation.yaml",
            "data_manipulation_with_different_connections.yaml",
        ],
        sampling_weights={
            "data_manipulation.yaml": 0.7,
            "data_manipulation_with_different_connections.yaml": 0.3,
        },
    ),
    CurriculumStage(
        name="stage_2",
        start_update=800,
        end_update=1199,
        scenario_names=[
            "data_manipulation.yaml",
            "data_manipulation_with_different_connections.yaml",
            "data_manipulation_plus_one_client.yaml",
        ],
        sampling_weights={
            "data_manipulation.yaml": 0.5,
            "data_manipulation_with_different_connections.yaml": 0.2,
            "data_manipulation_plus_one_client.yaml": 0.3,
        },
    ),
    CurriculumStage(
        name="stage_3",
        start_update=1200,
        end_update=1599,
        scenario_names=[
            "data_manipulation.yaml",
            "data_manipulation_with_different_connections.yaml",
            "data_manipulation_plus_one_client.yaml",
            "data_manipulation_plus_one_dynamic_client.yaml",
        ],
        sampling_weights={
            "data_manipulation.yaml": 0.4,
            "data_manipulation_with_different_connections.yaml": 0.15,
            "data_manipulation_plus_one_client.yaml": 0.2,
            "data_manipulation_plus_one_dynamic_client.yaml": 0.25,
        },
    ),
    CurriculumStage(
        name="stage_4",
        start_update=1600,
        end_update=None,
        scenario_names=[
            "data_manipulation.yaml",
            "data_manipulation_with_different_connections.yaml",
            "data_manipulation_plus_one_client.yaml",
            "data_manipulation_plus_one_dynamic_client.yaml",
            "data_manipulation_plus_two_dynamic_clients.yaml",
        ],
        sampling_weights={
            "data_manipulation.yaml": 0.35,
            "data_manipulation_with_different_connections.yaml": 0.10,
            "data_manipulation_plus_one_client.yaml": 0.15,
            "data_manipulation_plus_one_dynamic_client.yaml": 0.20,
            "data_manipulation_plus_two_dynamic_clients.yaml": 0.20,
        },
    ),
]


def validate_curriculum_stages(
    curriculum_stages: list[CurriculumStage],
    known_scenarios: set[str],
) -> list[CurriculumStage]:
    ordered = sorted(curriculum_stages, key=lambda row: row.start_update)

    if not ordered:
        raise ValueError("At least one curriculum stage is required.")
    if ordered[0].start_update != 0:
        raise ValueError("The first curriculum stage must start at update 0.")

    for idx, stage in enumerate(ordered):
        if not stage.scenario_names:
            raise ValueError(f"Stage '{stage.name}' has no scenarios.")

        for scenario_name in stage.scenario_names:
            if scenario_name not in known_scenarios:
                raise KeyError(
                    f"Unknown scenario '{scenario_name}' in stage '{stage.name}'."
                )

        if stage.end_update is not None and stage.end_update < stage.start_update:
            raise ValueError(f"Stage '{stage.name}' has end_update < start_update.")

        if idx > 0:
            prev_end = ordered[idx - 1].end_update
            if prev_end is None or stage.start_update != prev_end + 1:
                raise ValueError(
                    "Curriculum stages must be contiguous and only the final stage may be open ended."
                )

    if ordered[-1].end_update is not None:
        raise ValueError("The final curriculum stage must have end_update=None.")

    return ordered


def get_active_curriculum_stage(
    update: int,
    curriculum_stages: list[CurriculumStage],
) -> CurriculumStage:
    for stage in curriculum_stages:
        if update < stage.start_update:
            continue
        if stage.end_update is None or update <= stage.end_update:
            return stage
    raise ValueError(f"No curriculum stage found for update={update}.")


def sample_training_scenario(
    train_envs_by_name: dict[str, Any],
    active_scenario_names: list[str],
    sampling_weights: Optional[dict[str, float]] = None,
) -> tuple[str, Any]:
    active_scenario_names = list(active_scenario_names)

    if sampling_weights is None:
        probs = np.ones(len(active_scenario_names), dtype=np.float64)
    else:
        probs = np.asarray(
            [float(sampling_weights.get(name, 0.0)) for name in active_scenario_names],
            dtype=np.float64,
        )
        if float(probs.sum()) <= 0:
            raise ValueError("Scenario sampling weights must sum to a positive value.")

    probs = probs / probs.sum()
    idx = int(np.random.choice(len(active_scenario_names), p=probs))
    scenario_name = active_scenario_names[idx]
    return scenario_name, train_envs_by_name[scenario_name]


@torch.no_grad()
def evaluate_policy_on_env(
    agent,
    eval_env,
    device: torch.device,
    num_episodes: int = 5,
    max_steps: int = 128,
    greedy: bool = True,
) -> dict[str, Any]:
    returns: list[float] = []

    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            action_mask = eval_env.get_mask()
            node_mask = eval_env.get_node_mask(action_mask)
            obs_device = obs.clone().to(device)

            if greedy:
                node_type, node_local_idx, _, selected_emb, _ = agent.select_node(
                    agent.forward(obs_device), node_mask, greedy=True
                )
                action_local_idx, _, _ = agent.select_action(
                    node_type,
                    node_local_idx,
                    selected_emb,
                    action_mask,
                    greedy=True,
                )
                action = (node_type, node_local_idx, action_local_idx)
            else:
                action = agent.sample_action(obs_device, action_mask, node_mask)[0]

            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += reward

            if terminated or truncated:
                break

        returns.append(ep_return)

    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "all_returns": returns,
    }


@torch.no_grad()
def evaluate_agent_on_envs(
    agent,
    eval_envs_by_name: dict[str, Any],
    device: torch.device,
    scenario_order: list[str],
    num_episodes: int = 5,
    max_steps: int = 128,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for scenario_name in scenario_order:
        results[scenario_name] = evaluate_policy_on_env(
            agent=agent,
            eval_env=eval_envs_by_name[scenario_name],
            device=device,
            num_episodes=num_episodes,
            max_steps=max_steps,
            greedy=True,
        )

    return results


import matplotlib.pyplot as plt


def save_curriculum_plots(
    train_log: list[dict[str, Any]],
    eval_log: list[dict[str, Any]],
    scenario_order: list[str],
    plot_dir: str | Path,
) -> None:
    if not train_log:
        return

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_updates = [row["update"] for row in train_log]
    sampled_train_returns = [row["sampled_train_return"] for row in train_log]
    policy_losses = [row["policy_loss"] for row in train_log]
    value_losses = [row["value_loss"] for row in train_log]
    entropies = [row["entropy"] for row in train_log]
    explained_variances = [row["explained_variance"] for row in train_log]

    eval_updates = [row["update"] for row in eval_log]
    mean_eval_returns = [row["mean_return_all_scenarios"] for row in eval_log]

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(train_updates, sampled_train_returns, label="Sampled train return")
    if eval_log:
        ax1.plot(eval_updates, mean_eval_returns, label="Mean eval return")
    ax1.set_title("Training / eval returns")
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Return")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(train_updates, policy_losses, label="Policy loss")
    ax2.plot(train_updates, value_losses, label="Value loss")
    ax2.set_title("PPO losses")
    ax2.set_xlabel("Update")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(train_updates, entropies, label="Entropy")
    ax3.plot(train_updates, explained_variances, label="Explained variance")
    ax3.set_title("Entropy / value fit")
    ax3.set_xlabel("Update")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    for scenario_name in scenario_order:
        ax4.plot(
            eval_updates,
            [row["per_scenario"][scenario_name]["mean"] for row in eval_log],
            label=scenario_name.replace(".yaml", ""),
        )
    ax4.set_title("Per-scenario greedy eval")
    ax4.set_xlabel("Update")
    ax4.set_ylabel("Return")
    ax4.grid(True)
    if eval_log:
        ax4.legend(loc="best")

    fig.tight_layout()
    fig.savefig(plot_dir / "Multi_scenario_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_multi_scenario_curriculum(
    agent,
    optimizer: optim.Optimizer,
    train_cfgs_by_name: dict[str, dict[str, Any]],
    curriculum_stages: list[CurriculumStage],
    action_schema: dict[str, Any],
    device: torch.device,
    checkpoint_dir: str | Path,
    plot_dir: str | Path,
    scenario_order: list[str],
    total_updates: int = 2000,
    steps_per_rollout: int = 1024,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    learning_rate: float = 3e-4,
    learning_rate_end: float = 3e-5,
    lr_anneal: bool = True,
    entropy_start: float = 0.01,
    entropy_end: float = 0.001,
    target_kl: Optional[float] = 0.02,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_every: int = 10,
    save_every: int = 100,
    greedy_eval_episodes: int = 5,
    max_eval_steps: int = 128,
    train_envs_by_name: Optional[dict[str, Any]] = None,
    eval_envs_by_name: Optional[dict[str, Any]] = None,
    scenario_paths_by_name: Optional[dict[str, Path]] = None,
    resume_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    plot_dir = Path(plot_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoint_dir / "best.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"

    if train_envs_by_name is None:
        train_envs_by_name = {
            name: make_graph_env_from_cfg(cfg, action_schema)
            for name, cfg in train_cfgs_by_name.items()
        }
    if eval_envs_by_name is None:
        eval_envs_by_name = {
            name: make_graph_env_from_cfg(cfg, action_schema)
            for name, cfg in train_cfgs_by_name.items()
        }

    curriculum_metadata = {
        "scenario_names": list(scenario_order),
        "scenario_paths_by_name": {
            name: str(path) for name, path in (scenario_paths_by_name or {}).items()
        },
        "curriculum_stages": [asdict(stage) for stage in curriculum_stages],
        "include_packaged_in_schema": False,
    }

    train_log = (
        list(resume_payload.get("train_log", [])) if resume_payload is not None else []
    )
    eval_log = (
        list(resume_payload.get("eval_log", [])) if resume_payload is not None else []
    )

    scenario_rollout_counts = Counter()
    scenario_step_counts = Counter()

    if train_log:
        last_counts = train_log[-1].get("scenario_counts_so_far", {})
        scenario_rollout_counts.update(last_counts)
    if train_log:
        last_step_counts = train_log[-1].get("scenario_step_counts_so_far", {})
        scenario_step_counts.update(last_step_counts)

    best_mean_eval_return = -float("inf")
    if resume_payload is not None:
        best_mean_eval_return = float(
            (resume_payload.get("extra") or {}).get(
                "best_mean_eval_return", -float("inf")
            )
        )

    start_update = (
        int(resume_payload["update"]) + 1 if resume_payload is not None else 0
    )

    if start_update >= total_updates:
        save_curriculum_plots(train_log, eval_log, scenario_order, plot_dir)
        save_checkpoint(
            final_checkpoint_path,
            agent,
            optimizer,
            int(resume_payload["update"]),
            train_log,
            action_schema=action_schema,
            extra={
                "type": "final",
                "best_mean_eval_return": best_mean_eval_return,
                "curriculum_metadata": curriculum_metadata,
                "eval_log": eval_log,
            },
        )
        return {
            "train_log": train_log,
            "eval_log": eval_log,
            "best_checkpoint_path": best_checkpoint_path,
            "final_checkpoint_path": final_checkpoint_path,
            "best_mean_eval_return": best_mean_eval_return,
        }

    for update in range(start_update, total_updates):
        progress = update / max(total_updates - 1, 1)
        current_ent_coef = linear_schedule(entropy_start, entropy_end, progress)
        current_lr = (
            linear_schedule(learning_rate, learning_rate_end, progress)
            if lr_anneal
            else learning_rate
        )
        set_optimizer_lr(optimizer, current_lr)

        stage = get_active_curriculum_stage(update, curriculum_stages)
        sampled_scenario_name, sampled_env = sample_training_scenario(
            train_envs_by_name=train_envs_by_name,
            active_scenario_names=stage.scenario_names,
            sampling_weights=stage.sampling_weights,
        )

        batch = collect_rollout(
            agent=agent,
            env=sampled_env,
            steps_per_rollout=steps_per_rollout,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            scenario_id=sampled_scenario_name,
        )

        scenario_rollout_counts[sampled_scenario_name] += 1
        for scenario_name, count in batch["scenario_counts"].items():
            scenario_step_counts[scenario_name] += int(count)

        metrics = ppo_update(
            agent=agent,
            optimizer=optimizer,
            batch=batch,
            device=device,
            clip_coef=clip_coef,
            ent_coef=current_ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            normalize_adv=True,
            target_kl=target_kl,
        )

        sampled_train_return = (
            float(np.mean(batch["episode_returns"]))
            if batch["episode_returns"]
            else float("nan")
        )

        did_eval = (update % eval_every == 0) or (update == total_updates - 1)
        mean_eval_all = float("nan")

        if did_eval:
            eval_results = evaluate_agent_on_envs(
                agent=agent,
                eval_envs_by_name=eval_envs_by_name,
                device=device,
                scenario_order=scenario_order,
                num_episodes=greedy_eval_episodes,
                max_steps=max_eval_steps,
            )
            mean_eval_all = float(
                np.mean([eval_results[name]["mean"] for name in scenario_order])
            )
            eval_log.append(
                {
                    "update": int(update),
                    "mean_return_all_scenarios": mean_eval_all,
                    "per_scenario": copy.deepcopy(eval_results),
                }
            )

        row = {
            "update": int(update),
            "active_stage": stage.name,
            "sampled_scenario": sampled_scenario_name,
            "active_scenarios": list(stage.scenario_names),
            "scenario_counts_so_far": dict(sorted(scenario_rollout_counts.items())),
            "scenario_step_counts_so_far": dict(sorted(scenario_step_counts.items())),
            "sampled_train_return": sampled_train_return,
            "mean_eval_return_all_scenarios": (
                mean_eval_all
                if did_eval
                else (
                    eval_log[-1]["mean_return_all_scenarios"]
                    if eval_log
                    else float("nan")
                )
            ),
            "entropy_coef": float(current_ent_coef),
            "learning_rate": float(current_lr),
            **metrics,
        }
        train_log.append(row)

        if (
            did_eval
            and np.isfinite(mean_eval_all)
            and mean_eval_all > best_mean_eval_return
        ):
            best_mean_eval_return = mean_eval_all
            save_checkpoint(
                best_checkpoint_path,
                agent,
                optimizer,
                update,
                train_log,
                action_schema=action_schema,
                extra={
                    "type": "best_mean_eval_return",
                    "best_mean_eval_return": best_mean_eval_return,
                    "curriculum_metadata": curriculum_metadata,
                    "eval_log": eval_log,
                },
            )

        if ((update + 1) % save_every == 0) or (update == total_updates - 1):
            save_checkpoint(
                checkpoint_dir / f"checkpoint_update_{update:04d}.pt",
                agent,
                optimizer,
                update,
                train_log,
                action_schema=action_schema,
                extra={
                    "type": "periodic",
                    "best_mean_eval_return": best_mean_eval_return,
                    "curriculum_metadata": curriculum_metadata,
                    "eval_log": eval_log,
                },
            )

        if (
            did_eval
            or ((update + 1) % save_every == 0)
            or (update == total_updates - 1)
        ):
            save_curriculum_plots(train_log, eval_log, scenario_order, plot_dir)

        if (update % 10 == 0) or (update == total_updates - 1):
            LOG.info(
                "Update %s/%s | stage=%s | sampled=%s | train_return=%s | mean_eval=%s",
                update + 1,
                total_updates,
                stage.name,
                sampled_scenario_name,
                format_metric(sampled_train_return),
                format_metric(mean_eval_all),
            )

    save_checkpoint(
        final_checkpoint_path,
        agent,
        optimizer,
        total_updates - 1,
        train_log,
        action_schema=action_schema,
        extra={
            "type": "final",
            "best_mean_eval_return": best_mean_eval_return,
            "curriculum_metadata": curriculum_metadata,
            "eval_log": eval_log,
        },
    )

    save_curriculum_plots(train_log, eval_log, scenario_order, plot_dir)

    return {
        "train_log": train_log,
        "eval_log": eval_log,
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
        "best_mean_eval_return": best_mean_eval_return,
    }


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

set_seed(DEFAULT_SEED)
device = resolve_device(None)

for scenario_name, scenario_path in SCENARIO_PATHS_BY_NAME.items():
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario: {scenario_path}")

CURRICULUM_STAGES = validate_curriculum_stages(
    CURRICULUM_STAGES,
    set(SCENARIO_FILE_ORDER),
)

TRAIN_CFGS_BY_NAME = {
    scenario_name: load_cfg(resolve_yaml_path(scenario_path))
    for scenario_name, scenario_path in SCENARIO_PATHS_BY_NAME.items()
}

print(f"Loaded {len(TRAIN_CFGS_BY_NAME)} curriculum scenarios:")
for scenario_name in SCENARIO_FILE_ORDER:
    print(" -", scenario_name)

print("Using device:", device)

GLOBAL_SCHEMA = build_universal_schema(
    primary_cfgs=[TRAIN_CFGS_BY_NAME[name] for name in SCENARIO_FILE_ORDER],
    include_packaged=False,
)

print("GLOBAL_SCHEMA action_dims:", GLOBAL_SCHEMA["action_dims"])
print("GLOBAL_SCHEMA obs_dims   :", GLOBAL_SCHEMA["obs_dims"])

TRAIN_ENVS_BY_NAME = {
    scenario_name: make_graph_env_from_cfg(
        TRAIN_CFGS_BY_NAME[scenario_name], GLOBAL_SCHEMA
    )
    for scenario_name in SCENARIO_FILE_ORDER
}

EVAL_ENVS_BY_NAME = {
    scenario_name: make_graph_env_from_cfg(
        TRAIN_CFGS_BY_NAME[scenario_name], GLOBAL_SCHEMA
    )
    for scenario_name in SCENARIO_FILE_ORDER
}

training_config = TrainingConfig(total_updates=2000)

agent = build_agent_from_schema(
    action_schema=GLOBAL_SCHEMA,
    hidden_dim=training_config.hidden_dim,
    device=device,
)

optimizer = optim.Adam(agent.parameters(), lr=training_config.learning_rate)

training_result = train_multi_scenario_curriculum(
    agent=agent,
    optimizer=optimizer,
    train_cfgs_by_name=TRAIN_CFGS_BY_NAME,
    curriculum_stages=CURRICULUM_STAGES,
    action_schema=GLOBAL_SCHEMA,
    device=device,
    checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    plot_dir=DEFAULT_PLOT_DIR,
    scenario_order=SCENARIO_FILE_ORDER,
    total_updates=2000,
    steps_per_rollout=training_config.steps_per_rollout,
    update_epochs=training_config.update_epochs,
    minibatch_size=training_config.minibatch_size,
    learning_rate=training_config.learning_rate,
    learning_rate_end=training_config.learning_rate_end,
    lr_anneal=training_config.lr_anneal,
    entropy_start=training_config.entropy_start,
    entropy_end=training_config.entropy_end,
    target_kl=training_config.target_kl,
    gamma=training_config.gamma,
    gae_lambda=training_config.gae_lambda,
    clip_coef=training_config.clip_coef,
    vf_coef=training_config.vf_coef,
    max_grad_norm=training_config.max_grad_norm,
    eval_every=10,
    save_every=100,
    greedy_eval_episodes=5,
    max_eval_steps=128,
    train_envs_by_name=TRAIN_ENVS_BY_NAME,
    eval_envs_by_name=EVAL_ENVS_BY_NAME,
    scenario_paths_by_name=SCENARIO_PATHS_BY_NAME,
)

training_result["best_checkpoint_path"], training_result["final_checkpoint_path"]

print("Best checkpoint :", training_result["best_checkpoint_path"])
print("Final checkpoint:", training_result["final_checkpoint_path"])
print("Train log rows   :", len(training_result["train_log"]))
print("Eval log rows    :", len(training_result["eval_log"]))
print("Best mean eval   :", training_result["best_mean_eval_return"])
