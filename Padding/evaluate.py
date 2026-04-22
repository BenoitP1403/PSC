"""
Module d'évaluation pour l'approche padding (MaskablePPO).

Collecte les métriques par épisode : récompense moyenne/std/min/max,
courbes par pas, décomposition du reward shaping, et visualisations.

Usage
-----
    # Évaluer un modèle padding
    from padding.evaluate import evaluate_model
    stats = evaluate_model(model, env, n_episodes=50)

    # Comparer padding vs GNN (stats GNN fournies de l'extérieur)
    from padding.evaluate import compare_approaches
    report = compare_approaches(
        padding_model, padding_env,
        gnn_stats=gnn_eval_stats,   # EvalStats produit par le module GNN
        n_episodes=50,
    )
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    """Metrics from a single evaluation episode."""

    total_reward: float = 0.0
    base_reward: float = 0.0
    shaping_reward: float = 0.0
    length: int = 0
    per_step_rewards: List[float] = field(default_factory=list)
    detection_count: int = 0
    interruption_count: int = 0
    penalty_count: int = 0


@dataclass
class EvalStats:
    """Aggregate statistics across multiple evaluation episodes."""

    approach: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    median_reward: float
    mean_length: float
    mean_base_reward: float
    mean_shaping_reward: float
    mean_detection_count: float
    mean_interruption_count: float
    mean_penalty_count: float
    per_episode: List[EpisodeResult] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "approach": self.approach,
            "n_episodes": self.n_episodes,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "median_reward": self.median_reward,
            "mean_length": self.mean_length,
            "mean_base_reward": self.mean_base_reward,
            "mean_shaping_reward": self.mean_shaping_reward,
            "mean_detection_count": self.mean_detection_count,
            "mean_interruption_count": self.mean_interruption_count,
            "mean_penalty_count": self.mean_penalty_count,
        }

    def summary(self) -> str:
        lines = [
            f"=== {self.approach} Evaluation ({self.n_episodes} episodes) ===",
            f"  Reward: {self.mean_reward:.2f} +/- {self.std_reward:.2f}  "
            f"[{self.min_reward:.2f}, {self.max_reward:.2f}]",
            f"  Median: {self.median_reward:.2f}",
            f"  Length:  {self.mean_length:.1f}",
            f"  Base reward:    {self.mean_base_reward:.2f}",
            f"  Shaping reward: {self.mean_shaping_reward:.2f}",
            f"  Detections:     {self.mean_detection_count:.1f}",
            f"  Interruptions:  {self.mean_interruption_count:.1f}",
            f"  Penalties:      {self.mean_penalty_count:.1f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def _run_episode_padding(model, env, deterministic: bool = True) -> EpisodeResult:
    """Run one evaluation episode with a MaskablePPO (padding) model."""
    result = EpisodeResult()
    obs, info = env.reset()
    done = False

    while not done:
        # get action mask
        masks = None
        if hasattr(env, "action_masks"):
            masks = env.action_masks()

        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        result.total_reward += reward
        result.length += 1
        result.per_step_rewards.append(float(reward))

        # accumulate shaping stats from info
        result.base_reward += info.get("base_reward", reward)
        result.shaping_reward += info.get("shaping_reward", 0.0)
        result.detection_count = int(info.get("shaping/detection_count", 0))
        result.interruption_count = int(
            info.get("shaping/interruption_count", 0)
        )
        result.penalty_count = int(info.get("shaping/penalty_count", 0))

    return result



def evaluate_model(
    model,
    env,
    n_episodes: int = 50,
    approach: str = "padding",
    deterministic: bool = True,
) -> EvalStats:
    """
    Évalue un modèle MaskablePPO (padding) sur un environnement.

    Parameters
    ----------
    model : MaskablePPO
    env : gymnasium.Env avec action_masks()
    n_episodes : int
    approach : str
        Nom de l'approche utilisé dans EvalStats (défaut "padding").
    deterministic : bool
        Si False, l'agent échantillonne sa distribution plutôt que
        de prendre l'action modale — utile pour moyenner sur plusieurs
        nœuds de départ stochastiques.

    Returns
    -------
    EvalStats
    """
    episodes: List[EpisodeResult] = []

    for _ in range(n_episodes):
        ep = _run_episode_padding(model, env, deterministic=deterministic)
        episodes.append(ep)

    rewards = np.array([e.total_reward for e in episodes])
    lengths = np.array([e.length for e in episodes])

    return EvalStats(
        approach=approach,
        n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        median_reward=float(np.median(rewards)),
        mean_length=float(np.mean(lengths)),
        mean_base_reward=float(np.mean([e.base_reward for e in episodes])),
        mean_shaping_reward=float(
            np.mean([e.shaping_reward for e in episodes])
        ),
        mean_detection_count=float(
            np.mean([e.detection_count for e in episodes])
        ),
        mean_interruption_count=float(
            np.mean([e.interruption_count for e in episodes])
        ),
        mean_penalty_count=float(
            np.mean([e.penalty_count for e in episodes])
        ),
        per_episode=episodes,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def evaluate_random(env, n_episodes: int = 50) -> EvalStats:
    """Evaluate a uniformly random policy (respecting action masks)."""
    episodes: List[EpisodeResult] = []
    for _ in range(n_episodes):
        result = EpisodeResult()
        obs, info = env.reset()
        done = False
        while not done:
            masks = None
            if hasattr(env, "action_masks"):
                masks = env.action_masks()
            if masks is not None:
                valid = np.where(masks)[0]
                action = np.random.choice(valid) if len(valid) > 0 else 0
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            result.total_reward += reward
            result.length += 1
        episodes.append(result)

    rewards = np.array([e.total_reward for e in episodes])
    lengths = np.array([e.length for e in episodes])
    return EvalStats(
        approach="random",
        n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        median_reward=float(np.median(rewards)),
        mean_length=float(np.mean(lengths)),
        mean_base_reward=0.0,
        mean_shaping_reward=0.0,
        mean_detection_count=0.0,
        mean_interruption_count=0.0,
        mean_penalty_count=0.0,
        per_episode=episodes,
    )


def evaluate_noop(env, n_episodes: int = 50) -> EvalStats:
    """Evaluate a no-op policy (always action 0)."""
    episodes: List[EpisodeResult] = []
    for _ in range(n_episodes):
        result = EpisodeResult()
        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            result.total_reward += reward
            result.length += 1
        episodes.append(result)

    rewards = np.array([e.total_reward for e in episodes])
    lengths = np.array([e.length for e in episodes])
    return EvalStats(
        approach="noop",
        n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        median_reward=float(np.median(rewards)),
        mean_length=float(np.mean(lengths)),
        mean_base_reward=0.0,
        mean_shaping_reward=0.0,
        mean_detection_count=0.0,
        mean_interruption_count=0.0,
        mean_penalty_count=0.0,
        per_episode=episodes,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_approaches(
    padding_model,
    padding_env,
    gnn_stats: "EvalStats",
    n_episodes: int = 50,
    include_baselines: bool = True,
    output_dir: Optional[str] = None,
    deterministic: bool = False,
) -> Dict[str, EvalStats]:
    """
    Comparaison padding vs GNN sur le même scénario.

    Le module padding n'a pas connaissance de l'architecture GNN.
    Les statistiques GNN sont calculées par le module GNN du partenaire
    et passées ici directement via le paramètre gnn_stats.

    Parameters
    ----------
    padding_model : MaskablePPO
    padding_env : gymnasium.Env avec action_masks()
    gnn_stats : EvalStats
        Résultats pré-calculés par le module GNN (gnn_adapter.py).
    n_episodes : int
    include_baselines : bool
        Si True, évalue aussi les politiques random et noop.
    output_dir : str, optionnel
        Si fourni, sauvegarde le graphique et le JSON ici.
    deterministic : bool
        Transmis à evaluate_model pour le padding.

    Returns
    -------
    Dict[str, EvalStats]
    """
    results: Dict[str, EvalStats] = {}

    print("Évaluation padding...")
    results["padding"] = evaluate_model(
        padding_model, padding_env, n_episodes,
        approach="padding", deterministic=deterministic,
    )
    print(results["padding"].summary())
    print()

    results["gnn"] = gnn_stats
    print(gnn_stats.summary())
    print()

    if include_baselines:
        print("Baseline aléatoire...")
        results["random"] = evaluate_random(padding_env, n_episodes)
        print(results["random"].summary())
        print()

        print("Baseline no-op...")
        results["noop"] = evaluate_noop(padding_env, n_episodes)
        print(results["noop"].summary())
        print()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_data = {k: v.as_dict() for k, v in results.items()}
        with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
            json.dump(json_data, f, indent=2)
        if HAS_MPL:
            _plot_comparison(results, output_dir)

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _plot_comparison(results: Dict[str, EvalStats], output_dir: str):
    """Bar chart comparing mean rewards across approaches."""
    names = list(results.keys())
    means = [results[n].mean_reward for n in names]
    stds = [results[n].std_reward for n in names]

    colours = {
        "padding": "#4c72b0",
        "gnn": "#55a868",
        "random": "#c44e52",
        "noop": "#8172b2",
    }
    bar_colours = [colours.get(n, "#999999") for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=bar_colours, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Approach Comparison", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # annotate bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 1,
            f"{m:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_chart.png"), dpi=150)
    plt.close(fig)
    print(f"  Chart saved to {output_dir}/comparison_chart.png")


def plot_reward_curves(
    results: Dict[str, EvalStats], output_path: str
):
    """
    Plot per-step reward curves averaged across episodes for each approach.
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping reward curve plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = {"padding": "#4c72b0", "gnn": "#55a868"}

    for name, stats in results.items():
        if not stats.per_episode:
            continue
        # pad per-step rewards to same length
        max_len = max(len(e.per_step_rewards) for e in stats.per_episode)
        matrix = np.zeros((len(stats.per_episode), max_len))
        for i, e in enumerate(stats.per_episode):
            matrix[i, : len(e.per_step_rewards)] = e.per_step_rewards

        mean_curve = np.mean(matrix, axis=0)
        std_curve = np.std(matrix, axis=0)
        steps = np.arange(max_len)

        colour = colours.get(name, "#999999")
        ax.plot(steps, mean_curve, label=name, color=colour)
        ax.fill_between(
            steps, mean_curve - std_curve, mean_curve + std_curve,
            alpha=0.15, color=colour,
        )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Per-Step Reward Curve", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
