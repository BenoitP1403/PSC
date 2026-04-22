"""
Reward shaping for PrimAITE environments.

Adds three intrinsic signals to combat no-op collapse:
  1. Detection bonus   -- reward scanning a genuinely compromised node.
  2. Interruption bonus -- reward stalling the attacker's kill-chain.
  3. Unnecessary-action penalty -- penalise acting on fully-healthy nodes.

Design principles vs. the original implementation:
  - Accesses ground-truth state through a *single* accessor helper so the
    wrapper is not tightly coupled to PrimAITE internals.
  - Works with *any* inner env (flat, Dict, or graph-wrapped) -- it only
    reads info dicts and the unwrapped PrimAITE env.
  - Tracks per-episode statistics in a clean dataclass.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium
import numpy as np


# ---------------------------------------------------------------------------
# Episode statistics
# ---------------------------------------------------------------------------
@dataclass
class EpisodeShapingStats:
    """Tracks reward-shaping events for a single episode."""
    detection_count: int = 0
    detection_total: float = 0.0
    interruption_count: int = 0
    interruption_total: float = 0.0
    penalty_count: int = 0
    penalty_total: float = 0.0
    total_shaping: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "shaping/detection_count": self.detection_count,
            "shaping/detection_total": self.detection_total,
            "shaping/interruption_count": self.interruption_count,
            "shaping/interruption_total": self.interruption_total,
            "shaping/penalty_count": self.penalty_count,
            "shaping/penalty_total": self.penalty_total,
            "shaping/total": self.total_shaping,
        }


# ---------------------------------------------------------------------------
# Ground-truth accessor
# ---------------------------------------------------------------------------
class PrimAITEStateAccessor:
    """
    Single point of access for ground-truth simulation state.

    Isolates all PrimAITE-internal attribute accesses here so the rest of
    the shaping logic never touches framework internals directly.
    """

    def __init__(self, env: gymnasium.Env):
        self._base_env = env.unwrapped  # PrimaiteGymEnv

    def refresh(self):
        """Call after env.step() to cache fresh state references."""
        self._game = self._base_env.game
        self._agent = self._base_env.agent
        self._network = self._game.simulation.network
        # Build hostname -> node object map (network.nodes is UUID-keyed)
        self._hostname_to_node = {
            node_obj.config.hostname: node_obj
            for node_obj in self._network.nodes.values()
        }

    # -- node health --------------------------------------------------------
    def get_node_health(self, node_name: str) -> Optional[str]:
        """Return software health state string for *node_name*, or None."""
        try:
            node = self._hostname_to_node.get(node_name)
            if node is None:
                return None
            for sw in node.software_manager.software.values():
                state = getattr(sw, "health_state", None)
                if state is not None:
                    return str(state.name)
            return None
        except AttributeError:
            return None

    def is_node_compromised(self, node_name: str) -> bool:
        health = self.get_node_health(node_name)
        return health is not None and health in ("COMPROMISED", "OVERWHELMED")

    def is_node_healthy(self, node_name: str) -> bool:
        health = self.get_node_health(node_name)
        return health is not None and health == "GOOD"

    # -- action interpretation ----------------------------------------------
    def decode_action(self, action: int) -> Tuple[str, Dict[str, Any]]:
        """Decode integer action into (action_name, params)."""
        try:
            return self._agent.action_manager.get_action(action)
        except (KeyError, IndexError):
            return ("UNKNOWN", {})

    def action_targets_node(self, action: int) -> Optional[str]:
        """Return the node_name targeted by *action*, or None."""
        _, params = self.decode_action(action)
        return params.get("node_name", None)

    def is_scan_action(self, action: int) -> bool:
        name, _ = self.decode_action(action)
        name_lower = name.lower()
        return "scan" in name_lower

    # -- attacker kill-chain ------------------------------------------------
    def get_attacker_stage(self) -> Optional[int]:
        """
        Return the attacker's current kill-chain stage as an integer
        (0 = earliest, higher = deeper), or None if unavailable.
        """
        try:
            for agent in self._game.scripted_agents.values():
                kc = getattr(agent, "kill_chain", None) or getattr(
                    agent, "current_stage", None
                )
                if kc is not None:
                    if hasattr(kc, "value"):
                        return int(kc.value)
                    return int(kc)
            return None
        except Exception:
            return None

    @property
    def max_attacker_stage(self) -> int:
        """Upper bound for kill-chain stage (for normalisation)."""
        return 6  # conservative default covering PrimAITE TAP agents


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class RewardShapingWrapper(gymnasium.Wrapper):
    """
    Wraps *any* PrimAITE-compatible gymnasium env and augments the reward
    with three intrinsic shaping signals.

    Parameters
    ----------
    env : gymnasium.Env
        Inner environment (can be flat, Dict-obs, or graph-wrapped).
    detection_bonus : float
        Reward for scanning a compromised node.
    interruption_bonus_max : float
        Maximum reward for stalling the attacker (scaled by stage).
    interruption_streak : int
        How many consecutive steps the attacker must be stalled before
        the bonus triggers.
    unnecessary_penalty : float
        Penalty (negative) for acting on a fully-healthy node.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        detection_bonus: float = 5.0,
        interruption_bonus_max: float = 3.0,
        interruption_streak: int = 3,
        unnecessary_penalty: float = -1.0,
    ):
        super().__init__(env)
        self._accessor = PrimAITEStateAccessor(env)
        self._det_bonus = detection_bonus
        self._int_bonus_max = interruption_bonus_max
        self._int_streak_thresh = interruption_streak
        self._unnec_penalty = unnecessary_penalty

        # per-episode tracking
        self._stats = EpisodeShapingStats()
        self._prev_stage: Optional[int] = None
        self._stall_counter: int = 0

    # -- gymnasium API ------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._accessor.refresh()
        self._stats = EpisodeShapingStats()
        self._prev_stage = self._accessor.get_attacker_stage()
        self._stall_counter = 0
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self._accessor.refresh()

        shaping = self._compute_shaping(action)
        self._stats.total_shaping += shaping

        info["base_reward"] = base_reward
        info["shaping_reward"] = shaping
        info.update(self._stats.as_dict())

        return obs, base_reward + shaping, terminated, truncated, info

    # -- action masking passthrough -----------------------------------------
    def action_masks(self) -> np.ndarray:
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()
        n = self.action_space.n if hasattr(self.action_space, "n") else 1
        return np.ones(n, dtype=bool)

    # -- shaping signals ----------------------------------------------------
    def _compute_shaping(self, action) -> float:
        total = 0.0
        total += self._detection_signal(action)
        total += self._interruption_signal()
        total += self._unnecessary_signal(action)
        return total

    def _detection_signal(self, action) -> float:
        """Reward for scanning a node that is genuinely compromised."""
        if not isinstance(action, (int, np.integer)):
            return 0.0
        if not self._accessor.is_scan_action(int(action)):
            return 0.0
        target = self._accessor.action_targets_node(int(action))
        if target is None:
            return 0.0
        if self._accessor.is_node_compromised(target):
            self._stats.detection_count += 1
            self._stats.detection_total += self._det_bonus
            return self._det_bonus
        return 0.0

    def _interruption_signal(self) -> float:
        """Reward for stalling the attacker at the same kill-chain stage."""
        stage = self._accessor.get_attacker_stage()
        if stage is None or self._prev_stage is None:
            self._prev_stage = stage
            return 0.0

        if stage <= self._prev_stage:
            self._stall_counter += 1
        else:
            self._stall_counter = 0

        self._prev_stage = stage

        if self._stall_counter >= self._int_streak_thresh and stage > 0:
            max_stage = self._accessor.max_attacker_stage
            scale = stage / max(max_stage, 1)
            bonus = self._int_bonus_max * scale
            self._stats.interruption_count += 1
            self._stats.interruption_total += bonus
            return bonus
        return 0.0

    def _unnecessary_signal(self, action) -> float:
        """Penalise acting on a node that is already fully healthy."""
        if not isinstance(action, (int, np.integer)):
            return 0.0
        # scan actions are never penalised (information gathering is always ok)
        if self._accessor.is_scan_action(int(action)):
            return 0.0
        target = self._accessor.action_targets_node(int(action))
        if target is None:
            return 0.0
        if self._accessor.is_node_healthy(target):
            self._stats.penalty_count += 1
            self._stats.penalty_total += self._unnec_penalty
            return self._unnec_penalty
        return 0.0

    # -- convenience --------------------------------------------------------
    @property
    def episode_stats(self) -> EpisodeShapingStats:
        return self._stats
