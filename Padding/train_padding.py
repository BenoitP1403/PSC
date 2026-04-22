"""
End-to-end training script for padded PrimAITE environments.

Trains MaskablePPO policies on PrimAITE with the padding approach.
Supports three training modes:
  - "static": no dynamic nodes, all nodes visible
  - "dynamic": fixed dynamic node schedule
  - "randomized": per-episode topology randomization

Usage
-----
    # Train static policy
    python -m padding.train_padding \
        --config ../Scenarios/data_manipulation_3_pc.yaml \
        --mode static \
        --total-timesteps 100000

    # Train with dynamic nodes
    python -m padding.train_padding \
        --config ../Scenarios/data_manipulation_3_pc.yaml \
        --mode dynamic \
        --total-timesteps 100000

    # Train with domain randomization
    python -m padding.train_padding \
        --config ../Scenarios/data_manipulation_3_pc.yaml \
        --mode randomized \
        --total-timesteps 100000

    # Evaluate trained model
    python -m padding.train_padding \
        --config ../Scenarios/data_manipulation_3_pc.yaml \
        --mode static \
        --eval-only \
        --model-path models/ppo_static.zip
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from padding.reward_shaping import RewardShapingWrapper
from .domain_randomization import DomainRandomizationWrapper, RandomizationConfig
from .dynamic_node_wrapper import DynamicNodeConfig, DynamicNodeWrapper
from .structured_obs_wrapper import StructuredObsWrapper

# Try to import PrimaiteGymEnv
try:
    from primaite.session.environment import PrimaiteGymEnv
except ImportError:
    raise ImportError(
        "PrimaiteGymEnv not found. Install PrimAITE: pip install -e .[rl]"
    )


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env(
    config_path: str,
    mode: str = "static",
    dynamic_nodes: Optional[Dict[str, DynamicNodeConfig]] = None,
    reward_shaping: bool = True,
    rank: int = 0,
    rand_config: Optional["RandomizationConfig"] = None,
) -> gymnasium.Env:
    """
    Create and wrap a PrimAITE environment.

    Parameters
    ----------
    config_path : str
        Path to PrimAITE scenario YAML file.
    mode : str
        One of "static", "dynamic", "randomized".
    dynamic_nodes : Dict[str, DynamicNodeConfig], optional
        Dynamic node configuration (used if mode != "static").
    reward_shaping : bool
        Whether to apply reward shaping.
    rank : int
        Worker rank (for parallel environments).
    rand_config : RandomizationConfig, optional
        Domain randomization config (used only when mode="randomized").
        If None, falls back to RandomizationConfig() defaults.
        IMPORTANT: callers should always pass an explicit config — the
        default has clients_per_subnet_range=(2,5) which can leave the
        agent with only 2 visible nodes and makes training collapse.

    Returns
    -------
    gymnasium.Env
        Wrapped environment ready for training.
    """
    # Create base environment
    env = PrimaiteGymEnv(config_path)

    # Wrap with StructuredObsWrapper
    env = StructuredObsWrapper(env)

    # Wrap with DynamicNodeWrapper if needed
    if mode in ("dynamic", "randomized"):
        if dynamic_nodes is None:
            dynamic_nodes = {}
        env = DynamicNodeWrapper(env, dynamic_nodes)

    # Wrap with DomainRandomizationWrapper if needed
    if mode == "randomized":
        # Use the caller-supplied config if provided; warn if falling back to default.
        if rand_config is None:
            import warnings
            warnings.warn(
                "make_env(mode='randomized') called without rand_config. "
                "Using RandomizationConfig() default which allows as few as 2 "
                "visible nodes — training will likely collapse. "
                "Pass an explicit rand_config (e.g. from compare_randomized_padding.RAND_CONFIG).",
                stacklevel=2,
            )
            rand_config = RandomizationConfig()
        env = DomainRandomizationWrapper(env, rand_config)

    # Wrap with RewardShapingWrapper
    if reward_shaping:
        env = RewardShapingWrapper(
            env,
            detection_bonus=5.0,
            interruption_bonus_max=3.0,
            unnecessary_penalty=-1.0,
        )

    # ActionMasker as outermost wrapper so MaskablePPO can find action_masks()
    env = ActionMasker(env, lambda e: e.action_masks())

    return env


def make_mask_fn(env: gymnasium.Env):
    """
    Create a mask function for MaskablePPO.

    The mask function takes observation and returns a boolean array of valid actions.
    """
    def mask_fn(obs):
        # Check if env has action_masks method
        if hasattr(env, "action_masks"):
            masks = env.action_masks()
            if masks is not None:
                return masks
        # Default: all actions valid
        return np.ones(env.action_space.n, dtype=bool)

    return mask_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    config_path: str,
    mode: str = "static",
    total_timesteps: int = 100_000,
    n_seeds: int = 3,
    dynamic_nodes: Optional[Dict[str, DynamicNodeConfig]] = None,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    output_dir: str = "models",
) -> Tuple[MaskablePPO, Dict]:
    """
    Train a MaskablePPO policy.

    Parameters
    ----------
    config_path : str
        Path to scenario YAML.
    mode : str
        "static", "dynamic", or "randomized".
    total_timesteps : int
        Total training timesteps.
    n_seeds : int
        Number of seeds to average over.
    dynamic_nodes : Dict[str, DynamicNodeConfig], optional
        Dynamic node configs.
    learning_rate : float
        PPO learning rate.
    batch_size : int
        Mini-batch size.
    n_steps : int
        Steps per rollout.
    output_dir : str
        Directory for model checkpoints.

    Returns
    -------
    Tuple[MaskablePPO, Dict]
        Trained model and training statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create environment
    env = make_env(
        config_path,
        mode=mode,
        dynamic_nodes=dynamic_nodes,
        reward_shaping=True,
    )

    # Create vectorized environment for parallel training
    def make_env_fn():
        return make_env(
            config_path,
            mode=mode,
            dynamic_nodes=dynamic_nodes,
            reward_shaping=True,
        )

    # For now, use single env; can extend to DummyVecEnv if needed
    # vec_env = DummyVecEnv([make_env_fn for _ in range(n_seeds)])

    # Create model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        verbose=1,
        tensorboard_log="./logs",
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=output_dir,
        name_prefix=f"ppo_{mode}",
    )

    # Train
    print(f"Training {mode} policy for {total_timesteps} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        log_interval=10,
    )

    # Save final model
    model_path = os.path.join(output_dir, f"ppo_{mode}_final")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()

    return model, {"timesteps": total_timesteps, "mode": mode}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: MaskablePPO,
    env: gymnasium.Env,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a trained policy.

    Parameters
    ----------
    model : MaskablePPO
        Trained policy.
    env : gymnasium.Env
        Environment to evaluate on.
    n_episodes : int
        Number of evaluation episodes.
    deterministic : bool
        Whether to use deterministic policy.

    Returns
    -------
    Dict[str, float]
        Episode statistics (mean reward, success rate, etc.).
    """
    episode_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 1000:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            # Check for success (if available in info)
            if done and info.get("success", False):
                successes += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 5 == 0:
            print(f"Evaluated {episode + 1}/{n_episodes} episodes")

    env.close()

    stats = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(successes / n_episodes),
        "n_episodes": n_episodes,
    }

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate padded PrimAITE policies."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to scenario YAML file.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["static", "dynamic", "randomized"],
        default="static",
        help="Training mode.",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps.",
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of seeds to average over.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per rollout.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for model checkpoints.",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, do not train.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model to evaluate (for eval-only mode).",
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )

    args = parser.parse_args()

    if args.eval_only:
        # Evaluation mode
        if args.model_path is None:
            raise ValueError("--model-path required for eval-only mode.")

        env = make_env(args.config, mode=args.mode)
        model = MaskablePPO.load(args.model_path, env=env)

        stats = evaluate(model, env, n_episodes=args.n_eval_episodes)
        print(f"\nEvaluation Results ({args.mode}):")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")

    else:
        # Training mode
        dynamic_nodes = {}  # Can be extended to load from config

        model, train_info = train(
            args.config,
            mode=args.mode,
            total_timesteps=args.total_timesteps,
            n_seeds=args.n_seeds,
            dynamic_nodes=dynamic_nodes,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            output_dir=args.output_dir,
        )

        # Evaluate after training
        env = make_env(args.config, mode=args.mode)
        stats = evaluate(model, env, n_episodes=args.n_eval_episodes)
        print(f"\nTraining Complete ({args.mode}):")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
