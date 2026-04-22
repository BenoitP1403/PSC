"""
UniversalPaddingWrapper — fixes obs and action sizes across scenarios.

Allows a MaskablePPO model trained on scenario A to be evaluated on
scenario B, even when A and B have different obs/action space sizes.

How it works:
  - Wraps the full padding stack (StructuredObsWrapper + RewardShaping etc.)
  - Pads observations with zeros to a fixed target size
  - Pads action masks with False to a fixed target action count
  - Actions beyond the inner env's action space are silently mapped to 0 (no-op)

Usage
-----
    # Measure sizes of all relevant scenarios:
    sizes = measure_scenario_sizes(yaml_paths)
    fixed_obs  = sizes["max_obs"]
    fixed_acts = sizes["max_acts"]

    # Wrap each scenario with the same fixed sizes:
    env = make_padding_env(yaml_path, ...)
    env = UniversalPaddingWrapper(env, fixed_obs, fixed_acts)
"""
from __future__ import annotations

from typing import List

import gymnasium
import numpy as np


class UniversalPaddingWrapper(gymnasium.Wrapper):
    """
    Pads observations and action masks to fixed sizes so a model trained on
    one scenario can be evaluated on another.

    Parameters
    ----------
    env : gymnasium.Env
        Inner environment (any padding stack whose obs is a flat Box).
    fixed_obs_size : int
        Target observation size. Obs smaller than this are zero-padded;
        obs larger are truncated (should not happen if sized correctly).
    fixed_action_size : int
        Target action space size. Actions beyond the inner env's space
        are masked as invalid and mapped to no-op (action 0) on step().
    """

    def __init__(
        self,
        env: gymnasium.Env,
        fixed_obs_size: int,
        fixed_action_size: int,
    ):
        super().__init__(env)
        self._fixed_obs = fixed_obs_size
        self._fixed_acts = fixed_action_size
        self._inner_acts = env.action_space.n

        # Override spaces — match bounds of the inner env (pad with same low/high)
        inner_space = env.observation_space
        inner_size = inner_space.shape[0]
        low  = np.full(fixed_obs_size, inner_space.low[0],  dtype=np.float32)
        high = np.full(fixed_obs_size, inner_space.high[0], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, shape=(fixed_obs_size,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(fixed_action_size)

    # -- gymnasium API -------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_obs(obs), info

    def step(self, action):
        # map out-of-range actions to no-op
        if int(action) >= self._inner_acts:
            action = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pad_obs(obs), reward, terminated, truncated, info

    # -- action masking passthrough ------------------------------------------

    def action_masks(self) -> np.ndarray:
        inner_mask = None
        if hasattr(self.env, "action_masks"):
            inner_mask = self.env.action_masks()
        if inner_mask is None:
            inner_mask = np.ones(self._inner_acts, dtype=bool)
        # pad with False (extra actions masked) or truncate (out-of-range actions dropped)
        padded = np.zeros(self._fixed_acts, dtype=bool)
        n = min(len(inner_mask), self._fixed_acts)
        padded[:n] = inner_mask[:n]
        return padded

    # -- helpers -------------------------------------------------------------

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] == self._fixed_obs:
            return obs
        padded = np.zeros(self._fixed_obs, dtype=np.float32)
        n = min(obs.shape[0], self._fixed_obs)
        padded[:n] = obs[:n]
        return padded


# ---------------------------------------------------------------------------
# Helper: measure obs/action sizes across a list of scenarios
# ---------------------------------------------------------------------------

def measure_scenario_sizes(yaml_paths: List[str]) -> dict:
    """
    Instantiate each scenario briefly, record obs and action space sizes,
    return the maximums.

    Returns
    -------
    dict with keys:
        "max_obs"   : int
        "max_acts"  : int
        "per_scenario": list of {"yaml": str, "obs": int, "acts": int}
    """
    import yaml as _yaml
    from padding.train_padding import make_env as make_padding_env

    per_scenario = []
    for yaml_path in yaml_paths:
        # disable logging and ensure flatten_obs=True for the padding stack
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        for key in ("save_agent_actions", "save_step_metadata", "save_pcap_logs", "save_sys_logs"):
            cfg["io_settings"][key] = False
        for agent in cfg.get("agents", []):
            if agent.get("team") == "BLUE":
                agent.setdefault("agent_settings", {})["flatten_obs"] = True
        nolog_path = yaml_path.replace(".yaml", "_nolog.yaml")
        with open(nolog_path, "w") as f:
            _yaml.safe_dump(cfg, f, sort_keys=False)
        env = make_padding_env(nolog_path, mode="static", reward_shaping=False)
        obs_size = env.observation_space.shape[0]
        act_size = env.action_space.n
        env.close()
        per_scenario.append({"yaml": yaml_path, "obs": obs_size, "acts": act_size})
        print(f"  {yaml_path.split('/')[-1]}: obs={obs_size}, acts={act_size}")

    max_obs  = max(s["obs"]  for s in per_scenario)
    max_acts = max(s["acts"] for s in per_scenario)
    return {"max_obs": max_obs, "max_acts": max_acts, "per_scenario": per_scenario}
