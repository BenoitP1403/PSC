"""
Domain randomization wrapper for PrimAITE environments.

Design rationale:
  - Wraps a DynamicNodeWrapper initialized with a MAX-SIZE scenario
  - On each reset(), randomly decides which nodes are visible
  - Randomly determines when dynamic nodes become active
  - Enables robust training across diverse network topologies

The wrapper treats visibility/activation as a form of domain randomization:
  - Different subsets of nodes are visible each episode
  - Dynamic nodes activate at different times
  - This forces the policy to generalize across diverse topologies

Usage:
    env = PrimaiteGymEnv("path/to/max_scenario.yaml", flatten_obs=True)
    obs_wrapper = StructuredObsWrapper(env)

    dynamic_nodes = {f"node_{i}": DynamicNodeConfig(...) for i in range(100)}
    dynamic_wrapper = DynamicNodeWrapper(obs_wrapper, dynamic_nodes)

    config = RandomizationConfig(
        subnet_range=(1, 3),
        clients_per_subnet_range=(2, 5),
        dynamic_prob=0.3,
        activation_step_range=(20, 80),
    )
    randomizer = DomainRandomizationWrapper(dynamic_wrapper, config)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import gymnasium
import numpy as np

from .dynamic_node_wrapper import DynamicNodeConfig, DynamicNodeWrapper


@dataclass
class RandomizationConfig:
    """Configuration for per-episode domain randomization."""
    subnet_range: Tuple[int, int] = (1, 3)  # min/max subnets per episode
    clients_per_subnet_range: Tuple[int, int] = (2, 5)  # clients per subnet
    dynamic_prob: float = 0.3  # probability that a node is dynamic
    activation_step_range: Tuple[int, int] = (20, 80)  # activation step bounds


class DomainRandomizationWrapper(gymnasium.Wrapper):
    """
    Per-episode topology randomization for robust policy training.

    Wraps a DynamicNodeWrapper and randomizes on each reset():
      - Which nodes are visible
      - When dynamic nodes activate (if they become visible)
      - Attack parameters (for future extensions)

    Invisible nodes have:
      - presence = 0, announced = 0
      - observations zeroed
      - actions masked

    Parameters
    ----------
    env : gymnasium.Env
        Inner DynamicNodeWrapper environment (wrapping a max-size scenario).
    config : RandomizationConfig, optional
        Randomization settings. Defaults to RandomizationConfig().
    """

    def __init__(
        self,
        env: gymnasium.Env,
        config: Optional[RandomizationConfig] = None,
    ):
        super().__init__(env)

        if not isinstance(env, DynamicNodeWrapper):
            raise TypeError(
                f"DomainRandomizationWrapper requires DynamicNodeWrapper, got {type(env)}"
            )

        self._dynamic_wrapper: DynamicNodeWrapper = env
        self._config = config or RandomizationConfig()

        # Current episode's visible nodes
        self._visible_nodes: Set[str] = set()

    def _sample_topology(self) -> Set[str]:
        """
        Sample which nodes should be visible in this episode.

        Returns
        -------
        Set[str]
            Set of node names that should be visible.
        """
        node_registry = self._dynamic_wrapper._structured_wrapper.node_registry
        all_nodes = list(node_registry.keys())

        if not all_nodes:
            return set(all_nodes)

        # Randomly decide number of subnets
        num_subnets = np.random.randint(
            self._config.subnet_range[0],
            self._config.subnet_range[1] + 1,
        )

        # Distribute nodes across subnets
        nodes_per_subnet = np.random.randint(
            self._config.clients_per_subnet_range[0],
            self._config.clients_per_subnet_range[1] + 1,
            size=num_subnets,
        )

        total_visible = min(sum(nodes_per_subnet), len(all_nodes))
        visible = set(np.random.choice(all_nodes, size=total_visible, replace=False))

        return visible

    def _sample_dynamic_activations(
        self, visible_nodes: Set[str]
    ) -> Dict[str, DynamicNodeConfig]:
        """
        Sample activation times for dynamic nodes among the visible nodes.

        Returns
        -------
        Dict[str, DynamicNodeConfig]
            Activation configs for visible nodes that should be dynamic.
        """
        dynamic_configs: Dict[str, DynamicNodeConfig] = {}

        for node_name in visible_nodes:
            # Randomly decide if this node is dynamic
            if np.random.rand() < self._config.dynamic_prob:
                # Sample activation step
                activation_step = np.random.randint(
                    self._config.activation_step_range[0],
                    self._config.activation_step_range[1] + 1,
                )

                # Sample announcement lead
                announcement_lead = max(1, activation_step // 4)

                dynamic_configs[node_name] = DynamicNodeConfig(
                    activation_step=activation_step,
                    announcement_lead=announcement_lead,
                    action_indices=[],  # Will be populated by caller if needed
                    mode="scheduled",
                )

        return dynamic_configs

    # -- gymnasium API ------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment with a randomized topology.

        Samples visible nodes and their activation schedules, then delegates
        to the inner DynamicNodeWrapper.
        """
        # Sample topology for this episode
        self._visible_nodes = self._sample_topology()

        # Sample dynamic activations for visible nodes
        dynamic_configs = self._sample_dynamic_activations(self._visible_nodes)

        # Update inner dynamic wrapper's configuration
        # (This would require modifying DynamicNodeWrapper to support runtime updates,
        #  or we can achieve this by masking invisible nodes' obs and actions)
        self._dynamic_wrapper._dynamic_nodes = dynamic_configs
        self._dynamic_wrapper._reset_activation_steps()

        # Reset inner env
        obs, info = self.env.reset(seed=seed, options=options)

        # Mask invisible nodes
        obs = self._mask_invisible_nodes(obs)

        info["visible_nodes"] = list(self._visible_nodes)
        info["dynamic_nodes"] = list(dynamic_configs.keys())

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment and mask invisible nodes."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Mask invisible nodes (in case any became visible or announced)
        obs = self._mask_invisible_nodes(obs)

        return obs, reward, terminated, truncated, info

    # -- observation and action masking ------------------------------------

    def _mask_invisible_nodes(self, obs: np.ndarray) -> np.ndarray:
        """
        Zero out observations and set presence bits for invisible nodes.

        Invisible nodes get: present=0, announced=0, obs=0, actions masked.
        """
        obs = obs.copy()
        node_registry = self._dynamic_wrapper._structured_wrapper.node_registry

        for node_name, node_info in node_registry.items():
            if node_name not in self._visible_nodes:
                # Zero out this node's observation
                obs[node_info.obs_start : node_info.obs_end] = 0.0

                # Set presence/announced bits to 0
                self._dynamic_wrapper._structured_wrapper.set_node_presence(
                    node_name, present=False, announced=False
                )

        return obs

    def action_masks(self) -> np.ndarray:
        """Return action masks with invisible nodes masked."""
        masks = self.env.action_masks()
        if masks is None:
            masks = np.ones(self.action_space.n, dtype=bool)

        # Mask actions for invisible nodes
        dynamic_wrapper = self._dynamic_wrapper
        for node_name, config in dynamic_wrapper._dynamic_nodes.items():
            if node_name not in self._visible_nodes:
                for action_idx in config.action_indices:
                    if 0 <= action_idx < len(masks):
                        masks[action_idx] = False

        return masks

    # -- inspection ---------------------------------------------------------

    @property
    def visible_nodes(self) -> Set[str]:
        """Return current episode's visible nodes."""
        return set(self._visible_nodes)

    @property
    def all_nodes(self) -> Set[str]:
        """Return all available nodes."""
        return set(
            self._dynamic_wrapper._structured_wrapper.node_registry.keys()
        )

    def visibility_ratio(self) -> float:
        """Return fraction of nodes visible in current episode."""
        all_nodes = self.all_nodes
        if not all_nodes:
            return 1.0
        return len(self._visible_nodes) / len(all_nodes)
