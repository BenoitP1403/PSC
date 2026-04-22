"""
Structured observation wrapper for PrimAITE environments.

Design rationale:
  - Wraps a PrimaiteGymEnv (with flatten_obs=True) that provides flat observations
  - Augments observations with an explicit presence/announced mask
  - Each node gets 2 bits: [present, announced] indicating its activation state
  - Enables clean binary transitions (absent -> announced -> present)
  - Provides per-node observation slicing by name, avoiding fragile index arithmetic
  - Integrates with observation_manager to auto-discover node obs ranges

Usage:
    env = PrimaiteGymEnv(config_path, flatten_obs=True)
    obs_wrapper = StructuredObsWrapper(env)
    obs, info = obs_wrapper.reset()
    # obs is now: [flat_obs, presence_mask_node_1, announced_mask_node_1, ...]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium
import numpy as np


@dataclass
class NodeInfo:
    """Information about a node's observation range in flat space."""
    name: str
    obs_start: int  # inclusive
    obs_end: int    # exclusive
    obs_size: int


class StructuredObsWrapper(gymnasium.Wrapper):
    """
    Augments flat PrimAITE observations with per-node presence/announced bits.

    Observation space grows from (flat_size,) to (flat_size + 2*num_nodes,).
    The presence/announced bits are tracked as internal state and appended
    to each returned observation.

    Parameters
    ----------
    env : gymnasium.Env
        Inner PrimaiteGymEnv (must have flatten_obs=True).
    node_registry : Dict[str, NodeInfo], optional
        Pre-computed node info mapping. If None, auto-discover from env.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        node_registry: Optional[Dict[str, NodeInfo]] = None,
    ):
        super().__init__(env)

        # Verify we're wrapping a flat-obs environment
        if not isinstance(self.observation_space, gymnasium.spaces.Box):
            raise ValueError(
                f"StructuredObsWrapper requires flatten_obs=True. "
                f"Got observation_space type: {type(self.observation_space)}"
            )

        self._flat_obs_size = int(np.prod(self.observation_space.shape))

        # Build or validate node registry
        if node_registry is None:
            self._node_registry = self._build_node_registry()
        else:
            self._node_registry = node_registry

        # Mask state: node_name -> (present: bool, announced: bool)
        self._presence_state: Dict[str, Tuple[bool, bool]] = {}
        for node_name in self._node_registry.keys():
            self._presence_state[node_name] = (False, False)

        # Update observation space to include presence/announced masks
        num_nodes = len(self._node_registry)
        new_size = self._flat_obs_size + 2 * num_nodes
        self.observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(new_size,),
            dtype=np.float32,
        )

    # -- observation manipulation -------------------------------------------

    def _build_node_registry(self) -> Dict[str, NodeInfo]:
        """
        Introspect the environment to discover node observation ranges.

        Walks through env.unwrapped.agent.observation_manager to find
        which flat-obs indices correspond to which nodes.
        """
        try:
            unwrapped_env = self.env.unwrapped
            agent = unwrapped_env.agent
            obs_mgr = agent.observation_manager

            # Get the network to find all nodes
            game = unwrapped_env.game
            network = game.simulation.network
            node_names = list(network.nodes.keys())

            registry: Dict[str, NodeInfo] = {}

            # Try to get obs structure from observation_manager
            # This is environment-specific; adapt if the actual structure differs
            if hasattr(obs_mgr, "nodes_info"):
                # If obs_mgr provides structure, use it
                for node_name, info in obs_mgr.nodes_info.items():
                    start = getattr(info, "start_idx", 0)
                    end = getattr(info, "end_idx", 1)
                    registry[node_name] = NodeInfo(
                        name=node_name,
                        obs_start=start,
                        obs_end=end,
                        obs_size=end - start,
                    )
            else:
                # Fallback: assign equal-sized chunks per node
                chunk_size = max(1, self._flat_obs_size // len(node_names))
                for i, node_name in enumerate(sorted(node_names)):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, self._flat_obs_size)
                    registry[node_name] = NodeInfo(
                        name=node_name,
                        obs_start=start,
                        obs_end=end,
                        obs_size=end - start,
                    )

            return registry
        except Exception as e:
            raise RuntimeError(
                f"Failed to build node registry from environment: {e}"
            )

    def _append_presence_mask(self, flat_obs: np.ndarray) -> np.ndarray:
        """
        Append presence/announced bits to the flat observation.

        Returns observation of shape (flat_size + 2*num_nodes,).
        """
        mask_bits = []
        for node_name in sorted(self._node_registry.keys()):
            present, announced = self._presence_state[node_name]
            mask_bits.append(float(present))
            mask_bits.append(float(announced))

        mask = np.array(mask_bits, dtype=np.float32)
        return np.concatenate([flat_obs.astype(np.float32), mask])

    def _extract_flat_obs(self, obs: np.ndarray) -> np.ndarray:
        """Extract the flat portion from a structured obs."""
        return obs[: self._flat_obs_size]

    # -- gymnasium API ------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment and presence mask."""
        flat_obs, info = self.env.reset(seed=seed, options=options)

        # Reset all nodes to absent state
        for node_name in self._node_registry.keys():
            self._presence_state[node_name] = (False, False)

        return self._append_presence_mask(flat_obs), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment and append presence mask to observation."""
        flat_obs, reward, terminated, truncated, info = self.env.step(int(action))
        return (
            self._append_presence_mask(flat_obs),
            reward,
            terminated,
            truncated,
            info,
        )

    # -- action masking passthrough -----------------------------------------

    def action_masks(self) -> Optional[np.ndarray]:
        """Pass through action masks from inner env."""
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()
        return None

    # -- state management ---------------------------------------------------

    def set_node_presence(
        self, node_name: str, present: bool = False, announced: bool = False
    ) -> None:
        """
        Update presence/announced state for a node.

        Parameters
        ----------
        node_name : str
            Name of the node.
        present : bool
            True if the node is currently active/visible.
        announced : bool
            True if the node has been announced (but not yet present).
        """
        if node_name not in self._node_registry:
            raise ValueError(f"Unknown node: {node_name}")
        self._presence_state[node_name] = (present, announced)

    def get_node_presence(self, node_name: str) -> Tuple[bool, bool]:
        """Return (present, announced) state for a node."""
        if node_name not in self._node_registry:
            raise ValueError(f"Unknown node: {node_name}")
        return self._presence_state[node_name]

    # -- observation slicing ------------------------------------------------

    def get_node_obs(self, obs: np.ndarray, node_name: str) -> np.ndarray:
        """
        Extract the observation slice for a single node.

        Parameters
        ----------
        obs : np.ndarray
            Structured observation (returned by reset/step).
        node_name : str
            Name of the node.

        Returns
        -------
        np.ndarray
            Observation slice for that node (size = obs_size for that node).
        """
        if node_name not in self._node_registry:
            raise ValueError(f"Unknown node: {node_name}")

        info = self._node_registry[node_name]
        flat_obs = self._extract_flat_obs(obs)
        return flat_obs[info.obs_start : info.obs_end]

    def get_node_presence_bits(
        self, obs: np.ndarray, node_name: str
    ) -> Tuple[float, float]:
        """
        Extract presence/announced bits for a single node from observation.

        Parameters
        ----------
        obs : np.ndarray
            Structured observation.
        node_name : str
            Name of the node.

        Returns
        -------
        Tuple[float, float]
            (present_bit, announced_bit) as floats 0.0 or 1.0.
        """
        if node_name not in self._node_registry:
            raise ValueError(f"Unknown node: {node_name}")

        # Find the index of this node in sorted order
        sorted_names = sorted(self._node_registry.keys())
        node_idx = sorted_names.index(node_name)

        # Each node gets 2 bits starting at flat_obs_size + 2*node_idx
        start_idx = self._flat_obs_size + 2 * node_idx
        return float(obs[start_idx]), float(obs[start_idx + 1])

    # -- inspection ---------------------------------------------------------

    @property
    def node_registry(self) -> Dict[str, NodeInfo]:
        """Return the node registry."""
        return dict(self._node_registry)

    @property
    def presence_state(self) -> Dict[str, Tuple[bool, bool]]:
        """Return the current presence state (node_name -> (present, announced))."""
        return dict(self._presence_state)
