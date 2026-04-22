"""
Dynamic node activation wrapper for PrimAITE environments.

Design rationale:
  - Wraps a StructuredObsWrapper to manage node visibility over time
  - Supports scheduled (deterministic) and random activation modes
  - Implements announcement phase: nodes are announced (announced=1) before
    they become present (present=1)
  - Automatically masks actions and zeros observations for inactive nodes
  - Clean binary transitions: absent -> announced -> present

Usage:
    obs_wrapper = StructuredObsWrapper(env)
    dynamic_nodes = {
        "server_2": DynamicNodeConfig(
            activation_step=50,
            announcement_lead=5,
            action_indices=[2, 3, 4],  # actions specific to server_2
            mode="scheduled",
        )
    }
    dynamic_wrapper = DynamicNodeWrapper(obs_wrapper, dynamic_nodes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np

from .structured_obs_wrapper import StructuredObsWrapper


@dataclass
class DynamicNodeConfig:
    """Configuration for a single dynamic node."""
    activation_step: int  # timestep when node becomes present
    announcement_lead: int = 5  # steps before activation when node is announced
    action_indices: List[int] = field(default_factory=list)  # actions tied to this node
    mode: str = "scheduled"  # "scheduled" or "random"
    random_prob: float = 0.02  # probability of activation per step (if mode="random")


class DynamicNodeWrapper(gymnasium.Wrapper):
    """
    Manages dynamic node activation with announcement phases.

    Wraps a StructuredObsWrapper and controls when nodes become visible.
    Automatically handles:
      - Masking actions for inactive nodes
      - Zeroing observations for inactive nodes
      - Announcement phase (announced bit set, but obs still zeroed)
      - Clean state transitions

    Parameters
    ----------
    env : gymnasium.Env
        Inner StructuredObsWrapper environment.
    dynamic_nodes : Dict[str, DynamicNodeConfig]
        Configuration per dynamic node. Keys are node names.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        dynamic_nodes: Optional[Dict[str, DynamicNodeConfig]] = None,
    ):
        super().__init__(env)

        if not isinstance(env, StructuredObsWrapper):
            raise TypeError(
                f"DynamicNodeWrapper requires StructuredObsWrapper as inner env, "
                f"got {type(env)}"
            )

        self._structured_wrapper: StructuredObsWrapper = env
        self._dynamic_nodes = dynamic_nodes or {}

        # Validate that all dynamic node names exist
        for node_name in self._dynamic_nodes.keys():
            if node_name not in self._structured_wrapper.node_registry:
                raise ValueError(
                    f"Dynamic node '{node_name}' not in node registry. "
                    f"Available: {list(self._structured_wrapper.node_registry.keys())}"
                )

        # Track current step
        self._step_count = 0

        # Track activation state per node
        self._activation_steps: Dict[str, Optional[int]] = {}
        self._reset_activation_steps()

    def _reset_activation_steps(self) -> None:
        """Initialize activation steps for all dynamic nodes."""
        self._activation_steps = {}
        for node_name, config in self._dynamic_nodes.items():
            if config.mode == "scheduled":
                self._activation_steps[node_name] = config.activation_step
            elif config.mode == "random":
                self._activation_steps[node_name] = None  # will be decided at each step
            else:
                raise ValueError(f"Unknown mode: {config.mode}")

    # -- state queries ------------------------------------------------------

    def _is_node_announced(self, node_name: str, step: int) -> bool:
        """Check if node is in announcement phase."""
        if node_name not in self._dynamic_nodes:
            return False  # non-dynamic nodes are always "fully present"

        config = self._dynamic_nodes[node_name]
        act_step = self._activation_steps[node_name]

        if act_step is None:
            return False

        announcement_start = act_step - config.announcement_lead
        # Announced from announcement_start to (act_step - 1)
        return announcement_start <= step < act_step

    def _is_node_present(self, node_name: str, step: int) -> bool:
        """Check if node is fully present (past activation step)."""
        if node_name not in self._dynamic_nodes:
            return True  # non-dynamic nodes are always present

        act_step = self._activation_steps[node_name]
        if act_step is None:
            return False

        return step >= act_step

    # -- observation masking ------------------------------------------------

    def _mask_obs_for_inactive(self, obs: np.ndarray) -> np.ndarray:
        """
        Zero out observations for absent/announced nodes and update presence bits.

        For each dynamic node:
          - If absent: present=0, announced=0, obs=0
          - If announced: present=0, announced=1, obs=0
          - If present: present=1, announced=1, obs=real
        """
        obs = obs.copy()

        for node_name in self._dynamic_nodes.keys():
            is_announced = self._is_node_announced(node_name, self._step_count)
            is_present = self._is_node_present(node_name, self._step_count)

            # Zero out observation for inactive nodes
            if not is_present:
                node_info = self._structured_wrapper.node_registry[node_name]
                obs[node_info.obs_start : node_info.obs_end] = 0.0

            # Update presence bits
            self._structured_wrapper.set_node_presence(
                node_name, present=is_present, announced=is_announced or is_present
            )

        return obs

    # -- action masking -----------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Return action masks, masking out actions for inactive nodes.

        Passes through the inner action mask and then masks any actions
        that are tied to inactive dynamic nodes.
        """
        masks = self.env.action_masks()
        if masks is None:
            masks = np.ones(self.action_space.n, dtype=bool)

        for node_name, config in self._dynamic_nodes.items():
            is_present = self._is_node_present(node_name, self._step_count)

            if not is_present:
                # Mask out all actions for this node
                for action_idx in config.action_indices:
                    if 0 <= action_idx < len(masks):
                        masks[action_idx] = False

        return masks

    # -- dynamic activation -------------------------------------------------

    def _maybe_activate_nodes(self) -> None:
        """
        Handle random activation if any node uses mode='random'.

        For each random-mode node, sample whether to activate it at this step.
        """
        for node_name, config in self._dynamic_nodes.items():
            if config.mode != "random":
                continue

            # If already activated, skip
            if self._activation_steps[node_name] is not None:
                continue

            # Sample activation with configured probability
            if np.random.rand() < config.random_prob:
                self._activation_steps[node_name] = self._step_count

    # -- gymnasium API ------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment and dynamic node states."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._step_count = 0
        self._reset_activation_steps()

        # Mask observations for inactive nodes
        obs = self._mask_obs_for_inactive(obs)

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step environment, update dynamic node states, and mask observation.

        Step sequence:
          1. Call inner env.step()
          2. Check for random activations
          3. Mask observations and update presence bits
          4. Increment step counter
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Handle random activations
        self._maybe_activate_nodes()

        # Mask observations for inactive nodes
        obs = self._mask_obs_for_inactive(obs)

        self._step_count += 1
        info["dynamic_step_count"] = self._step_count

        return obs, reward, terminated, truncated, info

    # -- inspection ---------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Current step count."""
        return self._step_count

    @property
    def activation_steps(self) -> Dict[str, Optional[int]]:
        """Return activation steps per node."""
        return dict(self._activation_steps)

    def get_node_state(self, node_name: str) -> str:
        """
        Return the state of a node as a string.

        Returns one of: "absent", "announced", "present".
        """
        if node_name not in self._dynamic_nodes:
            return "present"  # non-dynamic nodes are always present

        if self._is_node_present(node_name, self._step_count):
            return "present"
        elif self._is_node_announced(node_name, self._step_count):
            return "announced"
        else:
            return "absent"

    def get_all_node_states(self) -> Dict[str, str]:
        """Return state for all dynamic nodes."""
        return {
            node_name: self.get_node_state(node_name)
            for node_name in self._dynamic_nodes.keys()
        }
