from __future__ import annotations
import argparse
import copy
import logging
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import matplotlib
import numpy as np
import platformdirs
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, HeteroConv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
PRIMAITE_SRC = PROJECT_ROOT / "PrimAITE" / "src"
WORKSPACE_TEMP = PROJECT_ROOT / ".tmp"
WORKSPACE_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(WORKSPACE_TEMP)
os.environ["TEMP"] = str(WORKSPACE_TEMP)
tempfile.tempdir = str(WORKSPACE_TEMP)


class _WorkspacePlatformDirs(platformdirs.PlatformDirs):

    @property
    def user_data_path(self) -> Path:
        path = PROJECT_ROOT / ".primaite_local" / str(self.appname) / str(self.version)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def user_config_path(self) -> Path:
        path = PROJECT_ROOT / ".primaite_user" / str(self.appname) / str(self.version)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def user_log_path(self) -> Path:
        path = self.user_data_path / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path


platformdirs.PlatformDirs = _WorkspacePlatformDirs

if PRIMAITE_SRC.exists() and str(PRIMAITE_SRC) not in sys.path:
    sys.path.insert(0, str(PRIMAITE_SRC))

import primaite
from primaite.session.environment import PrimaiteGymEnv


def redirect_primaite_runtime_paths() -> None:
    version = getattr(primaite, "__version__", "local")
    runtime_root = PROJECT_ROOT / ".primaite_runtime" / version

    primaite.PRIMAITE_PATHS.user_home_path = runtime_root
    primaite.PRIMAITE_PATHS.user_sessions_path = runtime_root / "sessions"
    primaite.PRIMAITE_PATHS.user_notebooks_path = runtime_root / "notebooks"

    primaite.PRIMAITE_PATHS.user_home_path.mkdir(parents=True, exist_ok=True)
    primaite.PRIMAITE_PATHS.user_sessions_path.mkdir(parents=True, exist_ok=True)
    primaite.PRIMAITE_PATHS.user_notebooks_path.mkdir(parents=True, exist_ok=True)


def configure_primaite_logging() -> None:
    primaite.PRIMAITE_CONFIG.setdefault("developer_mode", {})
    primaite.PRIMAITE_CONFIG["developer_mode"]["output_agent_logs"] = False
    primaite.PRIMAITE_CONFIG["developer_mode"]["output_sys_logs"] = False
    primaite.PRIMAITE_CONFIG["developer_mode"]["output_to_terminal"] = False
    primaite.PRIMAITE_CONFIG["developer_mode"]["agent_log_level"] = "WARNING"
    primaite.PRIMAITE_CONFIG["developer_mode"]["sys_log_level"] = "WARNING"

    from primaite.game.agent import agent_log as primaite_agent_log
    from primaite.simulator import LogLevel, SIM_OUTPUT

    SIM_OUTPUT.save_agent_logs = False
    SIM_OUTPUT.save_sys_logs = False
    SIM_OUTPUT.write_agent_log_to_terminal = False
    SIM_OUTPUT.write_sys_log_to_terminal = False
    SIM_OUTPUT.agent_log_level = LogLevel.WARNING
    SIM_OUTPUT.sys_log_level = LogLevel.WARNING

    for logger_name in ("primaite.session.environment", "primaite.session.io"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    try:
        from primaite.session import environment as primaite_environment
        from primaite.session import io as primaite_io

        primaite_environment._LOGGER.disabled = True
        primaite_io._LOGGER.disabled = True
    except Exception:
        pass

    original_setup_logger = primaite_agent_log.AgentLog.setup_logger

    def quiet_setup_logger(self) -> None:
        original_setup_logger(self)
        if getattr(self, "logger", None) is not None:
            self.logger.setLevel(logging.WARNING)
            self.logger.propagate = False

    primaite_agent_log.AgentLog.setup_logger = quiet_setup_logger


redirect_primaite_runtime_paths()
configure_primaite_logging()


LOG = logging.getLogger("gnn_single_scenario_ppo")

TARGET_MAP = {
    "node_name": "host",
    "target_router": "router",
    "target_firewall_nodename": "firewall",
}
TARGET_KEYS = tuple(TARGET_MAP.keys())
TARGET_KEY_BY_NODE_TYPE = {value: key for key, value in TARGET_MAP.items()}

PACKAGED_CFG_ROOT = Path("PrimAITE") / "src" / "primaite" / "config" / "_package_data"
UNIVERSAL_OBS_CAPS = {
    "host": 512,
    "router": 1024,
    "firewall": 1024,
    "link": 128,
    "switch": 1,
}

DEFAULT_TRAIN_SCENARIO = Path("../Scenarios/data_manipulation.yaml")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints_Single_Scenario_data_manipulation")
DEFAULT_PLOT_PATH = Path("training_plots") / "ppo_training_data_manipulation.png"


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 64
    total_updates: int = 2000
    steps_per_rollout: int = 1024
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    learning_rate_end: float = 3e-5
    lr_anneal: bool = True
    entropy_start: float = 0.01
    entropy_end: float = 0.001
    target_kl: Optional[float] = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_every: int = 10
    save_every: int = 100


DEFAULT_TRAINING_CONFIG = TrainingConfig()


def resolve_yaml_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YAML file not found: {resolved}")
    return resolved


def load_cfg(path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_yaml_path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("io_settings", {})
    cfg["io_settings"]["save_logs"] = False
    cfg["io_settings"]["save_agent_actions"] = False
    cfg["io_settings"]["save_step_metadata"] = False
    cfg["io_settings"]["save_pcap_logs"] = False
    cfg["io_settings"]["save_sys_logs"] = False
    cfg["io_settings"]["save_agent_logs"] = False
    cfg["io_settings"]["write_sys_log_to_terminal"] = False
    cfg["io_settings"]["write_agent_log_to_terminal"] = False
    cfg["io_settings"]["sys_log_level"] = "WARNING"
    cfg["io_settings"]["agent_log_level"] = "WARNING"

    for agent in cfg.get("agents", []):
        if agent.get("team") == "BLUE":
            agent.setdefault("agent_settings", {})
            agent["agent_settings"]["flatten_obs"] = False

    return cfg


def get_blue_agent_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    for agent in cfg.get("agents", []):
        if agent.get("team") == "BLUE":
            return agent
    raise ValueError("No BLUE agent found in configuration.")


class GraphWrapper(gym.Env):
    metadata = {"render_modes": ["graph"]}

    def __init__(
        self,
        primaite_env: PrimaiteGymEnv,
        shared_local_action_catalog: Optional[
            dict[str, list[tuple[str, dict[str, Any]]]]
        ] = None,
        shared_obs_dims: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()

        if primaite_env.agent.flatten_obs:
            raise ValueError("GraphWrapper requires non-flattened observations.")

        self._primaite_env = primaite_env
        self._warned_messages: set[str] = set()
        self._shared_local_action_catalog_override = copy.deepcopy(
            shared_local_action_catalog
        )
        self._shared_obs_dims_override = (
            copy.deepcopy(shared_obs_dims) if shared_obs_dims is not None else None
        )

        self._local_action_catalog, self._action_dims = (
            self._build_local_action_catalog()
        )
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ):
        obs, info = self._primaite_env.reset(seed=seed, options=options)
        return self._observation_change(obs), info

    def step(self, action: tuple[str, int, int]):
        node_type, node_local_idx, action_local_idx = action
        global_action = self._action_change(node_type, node_local_idx, action_local_idx)
        obs, reward, terminated, truncated, info = self._primaite_env.step(
            global_action
        )
        return self._observation_change(obs), reward, terminated, truncated, info

    def render(self) -> None:
        self._primaite_env.game.simulation.network.draw()

    def close(self) -> None:
        self._primaite_env.close()

    def get_obs_dims(self) -> dict[str, int]:
        return {
            node_type: int(np.prod(space.shape))
            for node_type, space in self.observation_space.spaces.items()
        }

    def get_raw_obs_dims(self) -> dict[str, int]:
        nodes = self._primaite_env.agent.observation_manager.obs.components["NODES"]
        links = self._primaite_env.agent.observation_manager.obs.components["LINKS"]

        dims = {"switch": 1}
        if nodes.hosts:
            dims["host"] = int(
                np.prod(gym.spaces.flatten_space(nodes.hosts[0].space).shape)
            )
        if nodes.routers:
            dims["router"] = int(
                np.prod(gym.spaces.flatten_space(nodes.routers[0].space).shape)
            )
        if nodes.firewalls:
            dims["firewall"] = int(
                np.prod(gym.spaces.flatten_space(nodes.firewalls[0].space).shape)
            )
        if links.links:
            dims["link"] = int(
                np.prod(gym.spaces.flatten_space(links.links[0].space).shape)
            )
        return dims

    def get_mask(self) -> dict[str, dict[int, torch.Tensor]]:
        global_action_mask = self._primaite_env.action_masks()
        nodes_by_type = self._get_nodes()
        action_map = self._primaite_env.agent.action_manager.action_map

        mask: dict[str, dict[int, torch.Tensor]] = {}
        for node_type, node_names in nodes_by_type.items():
            if node_type not in self._action_dims:
                continue
            mask[node_type] = {}
            for node_index in range(len(node_names)):
                node_mask = torch.zeros(self._action_dims[node_type], dtype=torch.bool)
                node_mask[0] = True
                mask[node_type][node_index] = node_mask

        for global_action_id, enabled in enumerate(global_action_mask):
            if enabled == 0 or global_action_id not in action_map:
                continue

            action_type, params = action_map[global_action_id]
            present_targets = [key for key in TARGET_KEYS if key in params] or list(
                TARGET_KEYS
            )
            filtered_params = {
                key: value for key, value in params.items() if key not in TARGET_KEYS
            }

            for target_key in present_targets:
                node_type = TARGET_MAP[target_key]
                local_idx = self._find_local_action_index(
                    node_type, action_type, filtered_params
                )
                if local_idx is None:
                    continue

                action_local_idx = local_idx + 1
                if target_key in params:
                    target_node_name = params[target_key]
                    if target_node_name in nodes_by_type.get(node_type, []):
                        node_local_idx = nodes_by_type[node_type].index(
                            target_node_name
                        )
                        mask[node_type][node_local_idx][action_local_idx] = True
                else:
                    for node_local_idx in mask.get(node_type, {}):
                        mask[node_type][node_local_idx][action_local_idx] = True

        return mask

    def get_node_mask(
        self, action_mask: dict[str, dict[int, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        node_mask: dict[str, torch.Tensor] = {}
        for node_type, nodes in action_mask.items():
            node_mask[node_type] = torch.tensor(
                [bool(mask.any().item()) for _, mask in sorted(nodes.items())],
                dtype=torch.bool,
            )
        return node_mask

    def _build_observation_space(self) -> gym.spaces.Dict:
        obs_dims = self._shared_obs_dims_override or self.get_raw_obs_dims()
        spaces = {
            node_type: gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dim,),
                dtype=np.float32,
            )
            for node_type, dim in obs_dims.items()
        }
        return gym.spaces.Dict(spaces)

    def _build_action_space(self) -> gym.spaces.Dict:
        spaces = {
            node_type: gym.spaces.Discrete(dim)
            for node_type, dim in self._action_dims.items()
        }
        return gym.spaces.Dict(spaces)

    def _build_local_action_catalog(
        self,
    ) -> tuple[dict[str, list[tuple[str, dict[str, Any]]]], dict[str, int]]:
        if self._shared_local_action_catalog_override is not None:
            catalog = {
                "host": list(
                    self._shared_local_action_catalog_override.get("host", [])
                ),
                "router": list(
                    self._shared_local_action_catalog_override.get("router", [])
                ),
                "firewall": list(
                    self._shared_local_action_catalog_override.get("firewall", [])
                ),
            }
            return catalog, action_dims_from_catalog(catalog)

        action_map = self._primaite_env.agent.action_manager.action_map
        catalog = {node_type: [] for node_type in TARGET_MAP.values()}
        seen = {node_type: set() for node_type in TARGET_MAP.values()}

        for _, (action_type, params) in action_map.items():
            present_targets = [key for key in TARGET_KEYS if key in params] or list(
                TARGET_KEYS
            )
            filtered_params = {
                key: value for key, value in params.items() if key not in TARGET_KEYS
            }
            frozen = (action_type, tuple(sorted(filtered_params.items())))

            for target_key in present_targets:
                node_type = TARGET_MAP[target_key]
                if frozen not in seen[node_type]:
                    seen[node_type].add(frozen)
                    catalog[node_type].append((action_type, filtered_params))

        action_dims = {
            node_type: 1 + len(actions) for node_type, actions in catalog.items()
        }
        return catalog, action_dims

    def _find_local_action_index(
        self,
        node_type: str,
        action_type: str,
        filtered_params: dict[str, Any],
    ) -> Optional[int]:
        for idx, (candidate_action_type, candidate_params) in enumerate(
            self._local_action_catalog[node_type]
        ):
            if (
                candidate_action_type == action_type
                and candidate_params == filtered_params
            ):
                return idx
        return None

    def _get_local_action(
        self, node_type: str, action_local_idx: int
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        if action_local_idx == 0:
            return None, None

        real_idx = action_local_idx - 1
        if real_idx < 0 or real_idx >= len(self._local_action_catalog[node_type]):
            raise IndexError(
                f"Invalid local action index {action_local_idx} for node type '{node_type}'."
            )
        return self._local_action_catalog[node_type][real_idx]

    def _action_change(
        self, node_type: str, node_local_idx: int, action_local_idx: int
    ) -> int:
        if action_local_idx == 0:
            return 0

        nodes_by_type = self._get_nodes()
        target_node_name = nodes_by_type[node_type][node_local_idx]
        action_type, filtered_params = self._get_local_action(
            node_type, action_local_idx
        )
        target_key = TARGET_KEY_BY_NODE_TYPE[node_type]

        action_map = self._primaite_env.agent.action_manager.action_map
        for global_action_id, (global_action_type, global_params) in action_map.items():
            if global_action_type != action_type:
                continue
            if target_key not in global_params:
                continue
            if global_params[target_key] != target_node_name:
                continue

            other_params = {
                key: value for key, value in global_params.items() if key != target_key
            }
            if other_params == filtered_params:
                return global_action_id

        LOG.warning(
            "Invalid action mapping for node_type=%s node_local_idx=%s action_local_idx=%s. "
            "Falling back to no-op.",
            node_type,
            node_local_idx,
            action_local_idx,
        )
        return 0

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_messages:
            return
        self._warned_messages.add(key)
        LOG.warning(message)

    def _fit_obs_vector(
        self, values: Any, target_dim: Optional[int], label: str
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if target_dim is None or values.shape[0] == target_dim:
            return values

        if values.shape[0] < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[: values.shape[0]] = values
            return padded

        self._warn_once(
            f"truncate:{label}:{values.shape[0]}->{target_dim}",
            f"Truncating {label} observation from {values.shape[0]} to {target_dim}.",
        )
        return values[:target_dim].astype(np.float32, copy=False)

    def _make_padded_feature(
        self, values: Any, target_dim: Optional[int], label: str
    ) -> tuple[np.ndarray, np.ndarray]:
        feature = self._fit_obs_vector(values, target_dim, label)
        feature_mask = np.ones(feature.shape[0], dtype=np.float32)
        raw_len = np.asarray(values, dtype=np.float32).reshape(-1).shape[0]
        if raw_len < feature.shape[0]:
            feature_mask[raw_len:] = 0.0
        return feature, feature_mask

    def _observation_change(self, observation: dict[str, Any]) -> HeteroData:
        declared_obs_dims = self.get_obs_dims()
        edge_list = self._get_edge_list()
        hosts_attr, routers_attr, firewalls_attr = self._get_nodes_attr(observation)
        link_attr = self._get_link_attr(observation)

        node_to_idx, _ = self._get_node_map()
        host_nodes = list(hosts_attr.keys())
        router_nodes = list(routers_attr.keys())
        firewall_nodes = list(firewalls_attr.keys())
        known_nodes = set(host_nodes) | set(router_nodes) | set(firewall_nodes)
        switch_nodes = [node for node in node_to_idx if node not in known_nodes]

        host_idx = {node: idx for idx, node in enumerate(host_nodes)}
        router_idx = {node: idx for idx, node in enumerate(router_nodes)}
        firewall_idx = {node: idx for idx, node in enumerate(firewall_nodes)}
        switch_idx = {node: idx for idx, node in enumerate(switch_nodes)}

        graph_data = HeteroData()

        if host_nodes:
            graph_data["host"].x = torch.stack(
                [
                    torch.tensor(hosts_attr[node]["x"], dtype=torch.float32)
                    for node in host_nodes
                ]
            )
            graph_data["host"].feat_mask = torch.stack(
                [
                    torch.tensor(hosts_attr[node]["feat_mask"], dtype=torch.float32)
                    for node in host_nodes
                ]
            )
        if router_nodes:
            graph_data["router"].x = torch.stack(
                [
                    torch.tensor(routers_attr[node]["x"], dtype=torch.float32)
                    for node in router_nodes
                ]
            )
            graph_data["router"].feat_mask = torch.stack(
                [
                    torch.tensor(routers_attr[node]["feat_mask"], dtype=torch.float32)
                    for node in router_nodes
                ]
            )
        if firewall_nodes:
            graph_data["firewall"].x = torch.stack(
                [
                    torch.tensor(firewalls_attr[node]["x"], dtype=torch.float32)
                    for node in firewall_nodes
                ]
            )
            graph_data["firewall"].feat_mask = torch.stack(
                [
                    torch.tensor(firewalls_attr[node]["feat_mask"], dtype=torch.float32)
                    for node in firewall_nodes
                ]
            )
        if switch_nodes:
            graph_data["switch"].x = torch.zeros(
                (len(switch_nodes), declared_obs_dims["switch"]),
                dtype=torch.float32,
            )
            graph_data["switch"].feat_mask = torch.ones(
                (len(switch_nodes), declared_obs_dims["switch"]),
                dtype=torch.float32,
            )

        def get_node_type_and_local_index(node_name: str) -> tuple[str, int]:
            if node_name in host_idx:
                return "host", host_idx[node_name]
            if node_name in router_idx:
                return "router", router_idx[node_name]
            if node_name in firewall_idx:
                return "firewall", firewall_idx[node_name]
            if node_name in switch_idx:
                return "switch", switch_idx[node_name]
            raise ValueError(f"Unknown node '{node_name}'.")

        rel_edges: dict[tuple[str, str, str], list[list[int]]] = {}
        rel_attr: dict[tuple[str, str, str], list[torch.Tensor]] = {}
        rel_attr_mask: dict[tuple[str, str, str], list[torch.Tensor]] = {}

        def add_edge(source_name: str, target_name: str) -> None:
            source_type, source_local = get_node_type_and_local_index(source_name)
            target_type, target_local = get_node_type_and_local_index(target_name)
            relation = (source_type, "link", target_type)

            key = (source_name, target_name)
            reverse_key = (target_name, source_name)
            if key in link_attr:
                payload = link_attr[key]
            elif reverse_key in link_attr:
                payload = link_attr[reverse_key]
            else:
                payload = {
                    "edge_attr": np.zeros(declared_obs_dims["link"], dtype=np.float32),
                    "edge_feat_mask": np.zeros(
                        declared_obs_dims["link"], dtype=np.float32
                    ),
                }
            rel_edges.setdefault(relation, []).append([source_local, target_local])
            rel_attr.setdefault(relation, []).append(
                torch.tensor(payload["edge_attr"], dtype=torch.float32)
            )
            rel_attr_mask.setdefault(relation, []).append(
                torch.tensor(payload["edge_feat_mask"], dtype=torch.float32)
            )

        for source_name, target_name in edge_list:
            add_edge(source_name, target_name)
            add_edge(target_name, source_name)

        for relation, edges in rel_edges.items():
            graph_data[relation].edge_index = (
                torch.tensor(edges, dtype=torch.long).t().contiguous()
            )
            graph_data[relation].edge_attr = torch.stack(rel_attr[relation], dim=0)
            graph_data[relation].edge_feat_mask = torch.stack(
                rel_attr_mask[relation], dim=0
            )

        self._validate_graph_data(graph_data)
        return graph_data

    def _validate_graph_data(self, graph_data: HeteroData) -> None:
        declared_obs_dims = self.get_obs_dims()

        for node_type in ("host", "router", "firewall", "switch"):
            if node_type not in graph_data.node_types:
                continue
            expected_dim = declared_obs_dims[node_type]
            actual_dim = int(graph_data[node_type].x.shape[-1])
            assert (
                actual_dim == expected_dim
            ), f"{node_type} feature width mismatch: {actual_dim} != {expected_dim}"
            if hasattr(graph_data[node_type], "feat_mask"):
                mask_dim = int(graph_data[node_type].feat_mask.shape[-1])
                assert (
                    mask_dim == expected_dim
                ), f"{node_type} feature mask width mismatch: {mask_dim} != {expected_dim}"

        expected_link_dim = declared_obs_dims["link"]
        for relation in graph_data.edge_types:
            actual_dim = int(graph_data[relation].edge_attr.shape[-1])
            assert (
                actual_dim == expected_link_dim
            ), f"{relation} edge_attr width mismatch: {actual_dim} != {expected_link_dim}"
            if hasattr(graph_data[relation], "edge_feat_mask"):
                mask_dim = int(graph_data[relation].edge_feat_mask.shape[-1])
                assert (
                    mask_dim == expected_link_dim
                ), f"{relation} edge feature mask width mismatch: {mask_dim} != {expected_link_dim}"

    def _get_nodes(self) -> dict[str, list[str]]:
        nodes = {"host": [], "router": [], "firewall": []}
        node_component = self._primaite_env.agent.observation_manager.obs.components[
            "NODES"
        ]

        for host in node_component.hosts:
            nodes["host"].append(host.where[-1])
        for router in node_component.routers:
            nodes["router"].append(router.where[-1])
        for firewall in node_component.firewalls:
            nodes["firewall"].append(firewall.where[-1])

        return nodes

    def _get_edge_list(self) -> list[tuple[str, str]]:
        graph = self._primaite_env.game.simulation.network._nx_graph
        return list(graph.edges())

    def _get_node_map(self) -> tuple[dict[str, int], dict[int, str]]:
        graph = self._primaite_env.game.simulation.network._nx_graph
        node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        idx_to_node = {idx: node for idx, node in enumerate(graph.nodes())}
        return node_to_idx, idx_to_node

    def _get_nodes_attr(self, observation: dict[str, Any]) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, dict[str, np.ndarray]],
        dict[str, dict[str, np.ndarray]],
    ]:
        hosts_attr: dict[str, dict[str, np.ndarray]] = {}
        routers_attr: dict[str, dict[str, np.ndarray]] = {}
        firewalls_attr: dict[str, dict[str, np.ndarray]] = {}

        node_component = self._primaite_env.agent.observation_manager.obs.components[
            "NODES"
        ]
        obs_dims = self.get_obs_dims()

        for idx, host in enumerate(node_component.hosts):
            flat = gym.spaces.flatten(host.space, observation["NODES"][f"HOST{idx}"])
            feat, feat_mask = self._make_padded_feature(
                flat, obs_dims.get("host"), "host"
            )
            hosts_attr[host.where[-1]] = {"x": feat, "feat_mask": feat_mask}

        for idx, router in enumerate(node_component.routers):
            flat = gym.spaces.flatten(
                router.space, observation["NODES"][f"ROUTER{idx}"]
            )
            feat, feat_mask = self._make_padded_feature(
                flat, obs_dims.get("router"), "router"
            )
            routers_attr[router.where[-1]] = {"x": feat, "feat_mask": feat_mask}

        for idx, firewall in enumerate(node_component.firewalls):
            flat = gym.spaces.flatten(
                firewall.space, observation["NODES"][f"FIREWALL{idx}"]
            )
            feat, feat_mask = self._make_padded_feature(
                flat,
                obs_dims.get("firewall"),
                "firewall",
            )
            firewalls_attr[firewall.where[-1]] = {"x": feat, "feat_mask": feat_mask}

        return hosts_attr, routers_attr, firewalls_attr

    def _get_link_attr(
        self, observation: dict[str, Any]
    ) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        links_attr: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        link_component = self._primaite_env.agent.observation_manager.obs.components[
            "LINKS"
        ]
        obs_dims = self.get_obs_dims()

        for idx, link in enumerate(link_component.links):
            endpoints = tuple(
                part.split(":")[0] for part in link.where[-1].split("<->")
            )
            flat = gym.spaces.flatten(link.space, observation["LINKS"][idx + 1])
            edge_attr, edge_feat_mask = self._make_padded_feature(
                flat, obs_dims.get("link"), "link"
            )
            links_attr[endpoints] = {
                "edge_attr": edge_attr,
                "edge_feat_mask": edge_feat_mask,
            }

        return links_attr


def extract_blue_action_map_from_cfg(
    cfg: dict[str, Any],
) -> dict[int, tuple[str, dict[str, Any]]]:
    blue_agent = get_blue_agent_cfg(cfg)
    raw_action_map = blue_agent["action_space"]["action_map"]
    action_map: dict[int, tuple[str, dict[str, Any]]] = {}

    for key, value in raw_action_map.items():
        action_map[int(key)] = (
            value["action"],
            copy.deepcopy(value.get("options", {})),
        )

    return action_map


def iter_packaged_cfg_paths(root: Path = PACKAGED_CFG_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.yaml") if path.is_file())


def collect_schema_cfgs(
    primary_cfgs: Sequence[dict[str, Any] | str | Path],
    include_packaged: bool = True,
) -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = []

    if include_packaged:
        for path in iter_packaged_cfg_paths():
            try:
                cfg = load_cfg(path)
                get_blue_agent_cfg(cfg)
                cfgs.append(cfg)
            except Exception as exc:
                LOG.debug("Skipping packaged schema config %s: %s", path, exc)

    for cfg in primary_cfgs:
        if cfg is None:
            continue
        if isinstance(cfg, (str, Path)):
            cfgs.append(load_cfg(cfg))
        else:
            cfgs.append(copy.deepcopy(cfg))

    return cfgs


def build_shared_local_action_catalog(
    cfgs: Sequence[dict[str, Any]],
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    catalog = {node_type: [] for node_type in TARGET_MAP.values()}
    seen = {node_type: set() for node_type in TARGET_MAP.values()}

    for cfg in cfgs:
        action_map = extract_blue_action_map_from_cfg(cfg)
        for _, (action_type, params) in action_map.items():
            if action_type == "do-nothing":
                continue

            present_targets = [key for key in TARGET_KEYS if key in params] or list(
                TARGET_KEYS
            )
            filtered_params = {
                key: value for key, value in params.items() if key not in TARGET_KEYS
            }
            frozen = (action_type, tuple(sorted(filtered_params.items())))

            for target_key in present_targets:
                node_type = TARGET_MAP[target_key]
                if frozen not in seen[node_type]:
                    seen[node_type].add(frozen)
                    catalog[node_type].append((action_type, filtered_params))

    for node_type in catalog:
        catalog[node_type] = sorted(
            catalog[node_type],
            key=lambda item: (item[0], str(sorted(item[1].items()))),
        )

    return catalog


def action_dims_from_catalog(
    shared_catalog: dict[str, list[tuple[str, dict[str, Any]]]],
) -> dict[str, int]:
    return {
        node_type: 1 + len(shared_catalog.get(node_type, []))
        for node_type in TARGET_MAP.values()
    }


def build_global_obs_dims(
    cfgs: Sequence[dict[str, Any]],
    obs_caps: Optional[dict[str, int]] = None,
    verbose: bool = True,
) -> dict[str, int]:
    obs_caps = dict(UNIVERSAL_OBS_CAPS if obs_caps is None else obs_caps)
    obs_dims = {"switch": 1, "host": 0, "router": 0, "firewall": 0, "link": 0}

    for idx, cfg in enumerate(cfgs):
        primaite_env: Optional[PrimaiteGymEnv] = None
        try:
            primaite_env = PrimaiteGymEnv(copy.deepcopy(cfg))
            env = GraphWrapper(primaite_env)
            raw_dims = env.get_raw_obs_dims()
            env.close()
            primaite_env = None

            for node_type, dim in raw_dims.items():
                obs_dims[node_type] = max(obs_dims.get(node_type, 0), dim)
        except Exception as exc:
            if verbose:
                LOG.warning(
                    "Skipping schema observation scan for cfg #%s: %s", idx, exc
                )
        finally:
            if primaite_env is not None:
                try:
                    primaite_env.close()
                except Exception:
                    pass

    for node_type, fallback_dim in obs_caps.items():
        if node_type == "switch":
            obs_dims[node_type] = 1
        elif obs_dims.get(node_type, 0) <= 0:
            obs_dims[node_type] = int(fallback_dim)

    return obs_dims


def build_universal_schema(
    primary_cfgs: Sequence[dict[str, Any] | str | Path],
    include_packaged: bool = True,
    obs_caps: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    LOG.info(
        "Building action and observation schema by scanning %s config families%s.",
        len(primary_cfgs),
        " plus packaged PrimAITE configs" if include_packaged else "",
    )
    cfgs = collect_schema_cfgs(primary_cfgs, include_packaged=include_packaged)
    shared_catalog = build_shared_local_action_catalog(cfgs)
    obs_dims = build_global_obs_dims(cfgs, obs_caps=obs_caps, verbose=True)
    return {
        "shared_catalog": shared_catalog,
        "action_dims": action_dims_from_catalog(shared_catalog),
        "obs_dims": obs_dims,
    }


def make_graph_env_from_cfg(
    cfg: dict[str, Any], action_schema: dict[str, Any]
) -> GraphWrapper:
    return GraphWrapper(
        primaite_env=PrimaiteGymEnv(copy.deepcopy(cfg)),
        shared_local_action_catalog=action_schema["shared_catalog"],
        shared_obs_dims=action_schema["obs_dims"],
    )


class GraphPPOAgent(nn.Module):
    def __init__(
        self, obs_dim: dict[str, int], action_dim: dict[str, int], hidden_dim: int = 64
    ):
        super().__init__()

        self.obs_dim = dict(obs_dim)
        self.action_dim = dict(action_dim)

        self.node_lin = nn.ModuleDict(
            {
                node_type: nn.Linear(2 * obs_width, hidden_dim)
                for node_type, obs_width in obs_dim.items()
                if node_type not in {"link", "switch"}
            }
        )
        self.edge_lin = nn.Linear(2 * obs_dim["link"], hidden_dim)
        self.switch_emb = nn.Embedding(1, hidden_dim)

        self.gine_conv1 = self._make_gine_conv(hidden_dim)
        self.gine_conv2 = self._make_gine_conv(hidden_dim)
        self.act = nn.ReLU()

        self.actor_node_head = nn.Linear(hidden_dim, 1)
        self.actor_action_heads = nn.ModuleDict(
            {
                node_type: nn.Linear(hidden_dim, dim)
                for node_type, dim in action_dim.items()
            }
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

    def _concat_node_features_and_mask(
        self, obs: HeteroData, node_type: str
    ) -> torch.Tensor:
        node_features = obs[node_type].x
        feature_mask = getattr(obs[node_type], "feat_mask", None)
        if feature_mask is None:
            feature_mask = torch.ones_like(node_features)
        return torch.cat([node_features, feature_mask], dim=-1)

    def _concat_edge_features_and_mask(
        self, obs: HeteroData, relation: tuple[str, str, str]
    ) -> torch.Tensor:
        edge_attr = obs[relation].edge_attr
        edge_feat_mask = getattr(obs[relation], "edge_feat_mask", None)
        if edge_feat_mask is None:
            edge_feat_mask = torch.ones_like(edge_attr)
        return torch.cat([edge_attr, edge_feat_mask], dim=-1)

    def forward(self, obs: HeteroData) -> dict[str, torch.Tensor]:
        x_dict: dict[str, torch.Tensor] = {}

        for node_type in obs.node_types:
            if node_type in self.node_lin:
                encoded = self._concat_node_features_and_mask(obs, node_type)
                x_dict[node_type] = self.act(self.node_lin[node_type](encoded))

        if "switch" in obs.node_types:
            num_switches = obs["switch"].num_nodes
            switch_index = torch.zeros(
                num_switches, dtype=torch.long, device=obs["switch"].x.device
            )
            x_dict["switch"] = self.act(self.switch_emb(switch_index))

        edge_attr_dict = {
            relation: self.edge_lin(self._concat_edge_features_and_mask(obs, relation))
            for relation in obs.edge_types
        }

        x_dict = self.gine_conv1(
            x_dict, obs.edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        x_dict = {node_type: self.act(x) for node_type, x in x_dict.items()}
        x_dict = self.gine_conv2(
            x_dict, obs.edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        x_dict = {node_type: self.act(x) for node_type, x in x_dict.items()}
        return x_dict

    def _build_flat_node_view(
        self, x_dict: dict[str, torch.Tensor]
    ) -> tuple[list[str], torch.Tensor, np.ndarray]:
        valid_types = [
            node_type for node_type in x_dict if node_type in self.actor_action_heads
        ]
        if not valid_types:
            raise ValueError("No selectable node types found in graph encoding.")

        flat_scores = torch.cat(
            [
                self.actor_node_head(x_dict[node_type]).squeeze(-1)
                for node_type in valid_types
            ],
            dim=0,
        )
        sizes = [x_dict[node_type].shape[0] for node_type in valid_types]
        cumulative_sizes = np.cumsum([0] + sizes)
        return valid_types, flat_scores, cumulative_sizes

    @staticmethod
    def _node_global_to_local(
        node_idx: int, valid_types: list[str], cumulative_sizes: np.ndarray
    ) -> tuple[str, int]:
        for idx, node_type in enumerate(valid_types):
            if cumulative_sizes[idx] <= node_idx < cumulative_sizes[idx + 1]:
                return node_type, node_idx - int(cumulative_sizes[idx])
        raise ValueError(f"Invalid flattened node index {node_idx}.")

    def select_node(
        self,
        x_dict: dict[str, torch.Tensor],
        node_mask: dict[str, torch.Tensor],
        greedy: bool = False,
    ) -> tuple[str, int, torch.Tensor, torch.Tensor, Categorical]:
        valid_types, flat_scores, cumulative_sizes = self._build_flat_node_view(x_dict)
        device = flat_scores.device
        flat_node_mask = torch.cat(
            [node_mask[node_type].to(device) for node_type in valid_types], dim=0
        )
        masked_scores = torch.where(
            flat_node_mask, flat_scores, torch.full_like(flat_scores, -1e9)
        )
        node_dist = Categorical(logits=masked_scores)

        if greedy:
            node_idx = int(torch.argmax(masked_scores).item())
        else:
            node_idx = int(node_dist.sample().item())

        logp_node = node_dist.log_prob(torch.as_tensor(node_idx, device=device))
        node_type, node_local_idx = self._node_global_to_local(
            node_idx, valid_types, cumulative_sizes
        )
        selected_emb = x_dict[node_type][node_local_idx]
        return node_type, node_local_idx, logp_node, selected_emb, node_dist

    def select_action(
        self,
        node_type: str,
        node_local_idx: int,
        selected_emb: torch.Tensor,
        action_mask: dict[str, dict[int, torch.Tensor]],
        greedy: bool = False,
    ) -> tuple[int, torch.Tensor, Categorical]:
        action_logits = self.actor_action_heads[node_type](selected_emb)
        device = action_logits.device
        mask = action_mask[node_type][node_local_idx].to(device)
        masked_logits = torch.where(
            mask, action_logits, torch.full_like(action_logits, -1e9)
        )
        action_dist = Categorical(logits=masked_logits)

        if greedy:
            action_local_idx = int(torch.argmax(masked_logits).item())
        else:
            action_local_idx = int(action_dist.sample().item())

        logp_action = action_dist.log_prob(
            torch.as_tensor(action_local_idx, device=device)
        )
        return action_local_idx, logp_action, action_dist

    def get_value(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        all_embeddings = torch.cat([embedding for embedding in x_dict.values()], dim=0)
        graph_embedding = all_embeddings.mean(dim=0)
        return self.critic_head(graph_embedding).squeeze(-1)

    def sample_action(
        self,
        obs: HeteroData,
        action_mask: dict[str, dict[int, torch.Tensor]],
        node_mask: dict[str, torch.Tensor],
    ) -> tuple[tuple[str, int, int], torch.Tensor, torch.Tensor, torch.Tensor]:
        x_dict = self.forward(obs)
        node_type, node_local_idx, logp_node, selected_emb, node_dist = (
            self.select_node(
                x_dict,
                node_mask,
                greedy=False,
            )
        )
        action_local_idx, logp_action, action_dist = self.select_action(
            node_type,
            node_local_idx,
            selected_emb,
            action_mask,
            greedy=False,
        )

        value = self.get_value(x_dict)
        total_logp = logp_node + logp_action
        entropy = node_dist.entropy() + action_dist.entropy()
        action = (node_type, node_local_idx, action_local_idx)
        return action, total_logp, entropy, value

    def evaluate_action(
        self,
        obs: HeteroData,
        action: tuple[str, int, int],
        action_mask: dict[str, dict[int, torch.Tensor]],
        node_mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_node_type, chosen_node_local_idx, chosen_action_local_idx = action
        x_dict = self.forward(obs)
        value = self.get_value(x_dict)

        valid_types, flat_scores, cumulative_sizes = self._build_flat_node_view(x_dict)
        device = flat_scores.device
        flat_node_mask = torch.cat(
            [node_mask[node_type].to(device) for node_type in valid_types], dim=0
        )
        masked_scores = torch.where(
            flat_node_mask, flat_scores, torch.full_like(flat_scores, -1e9)
        )
        node_dist = Categorical(logits=masked_scores)

        chosen_flat_idx = None
        for idx, node_type in enumerate(valid_types):
            if node_type == chosen_node_type:
                chosen_flat_idx = int(cumulative_sizes[idx]) + chosen_node_local_idx
                break
        if chosen_flat_idx is None:
            raise ValueError(f"Chosen node type '{chosen_node_type}' is not available.")

        logp_node = node_dist.log_prob(torch.as_tensor(chosen_flat_idx, device=device))

        selected_emb = x_dict[chosen_node_type][chosen_node_local_idx]
        action_logits = self.actor_action_heads[chosen_node_type](selected_emb)
        action_device = action_logits.device
        chosen_action_mask = action_mask[chosen_node_type][chosen_node_local_idx].to(
            action_device
        )
        masked_action_logits = torch.where(
            chosen_action_mask,
            action_logits,
            torch.full_like(action_logits, -1e9),
        )
        action_dist = Categorical(logits=masked_action_logits)
        logp_action = action_dist.log_prob(
            torch.as_tensor(chosen_action_local_idx, device=action_device)
        )

        total_logp = logp_node + logp_action
        entropy = node_dist.entropy() + action_dist.entropy()
        return total_logp, entropy, value

    @staticmethod
    def _make_gine_conv(hidden_dim: int) -> HeteroConv:
        relations = [
            ("switch", "link", "router"),
            ("router", "link", "switch"),
            ("switch", "link", "host"),
            ("host", "link", "switch"),
            ("switch", "link", "firewall"),
            ("firewall", "link", "switch"),
        ]
        return HeteroConv(
            {
                relation: GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                for relation in relations
            },
            aggr="sum",
        )


def build_agent_from_schema(
    action_schema: dict[str, Any],
    hidden_dim: int,
    device: torch.device,
) -> GraphPPOAgent:
    return GraphPPOAgent(
        obs_dim=action_schema["obs_dims"],
        action_dim=action_schema["action_dims"],
        hidden_dim=hidden_dim,
    ).to(device)


class RolloutBuffer:
    def __init__(self) -> None:
        self.obs: list[HeteroData] = []
        self.action_masks: list[dict[str, dict[int, torch.Tensor]]] = []
        self.node_masks: list[dict[str, torch.Tensor]] = []
        self.actions: list[tuple[str, int, int]] = []
        self.logprobs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []

    def add(
        self,
        obs: HeteroData,
        action_mask: dict[str, dict[int, torch.Tensor]],
        node_mask: dict[str, torch.Tensor],
        action: tuple[str, int, int],
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.obs.append(obs)
        self.action_masks.append(action_mask)
        self.node_masks.append(node_mask)
        self.actions.append(action)
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))


def compute_gae(
    rewards: Sequence[float],
    dones: Sequence[float],
    values: Sequence[float],
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)

    advantages = np.zeros_like(rewards_np, dtype=np.float32)
    last_gae = 0.0

    for timestep in reversed(range(len(rewards_np))):
        if timestep == len(rewards_np) - 1:
            next_nonterminal = 1.0 - dones_np[timestep]
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones_np[timestep]
            next_value = values_np[timestep + 1]

        delta = (
            rewards_np[timestep]
            + gamma * next_value * next_nonterminal
            - values_np[timestep]
        )
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[timestep] = last_gae

    returns = advantages + values_np
    return advantages, returns


@torch.no_grad()
def collect_rollout(
    agent: GraphPPOAgent,
    env: GraphWrapper,
    steps_per_rollout: int,
    device: torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    scenario_id: str = "train",
) -> dict[str, Any]:
    buffer = RolloutBuffer()
    scenario_counts: dict[str, int] = {}
    episode_returns: list[float] = []
    current_episode_return = 0.0

    obs, _ = env.reset()

    for _ in range(steps_per_rollout):
        action_mask = env.get_mask()
        node_mask = env.get_node_mask(action_mask)
        obs_device = obs.clone().to(device)

        action, logp, _, value = agent.sample_action(obs_device, action_mask, node_mask)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(
            obs=obs.clone(),
            action_mask=action_mask,
            node_mask=node_mask,
            action=action,
            logprob=logp.item(),
            reward=reward,
            done=done,
            value=value.item(),
        )

        scenario_counts[scenario_id] = scenario_counts.get(scenario_id, 0) + 1
        current_episode_return += reward
        obs = next_obs

        if done:
            episode_returns.append(current_episode_return)
            current_episode_return = 0.0
            obs, _ = env.reset()

    action_mask = env.get_mask()
    node_mask = env.get_node_mask(action_mask)
    last_value = agent.get_value(agent.forward(obs.clone().to(device))).item()

    advantages, returns = compute_gae(
        rewards=buffer.rewards,
        dones=buffer.dones,
        values=buffer.values,
        last_value=last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    return {
        "obs": buffer.obs,
        "action_masks": buffer.action_masks,
        "node_masks": buffer.node_masks,
        "actions": buffer.actions,
        "old_logprobs": torch.tensor(
            buffer.logprobs, dtype=torch.float32, device=device
        ),
        "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
        "returns": torch.tensor(returns, dtype=torch.float32, device=device),
        "scenario_counts": scenario_counts,
        "episode_returns": episode_returns,
    }


def ppo_update(
    agent: GraphPPOAgent,
    optimizer: optim.Optimizer,
    batch: dict[str, Any],
    device: torch.device,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    normalize_adv: bool = True,
    target_kl: Optional[float] = None,
) -> dict[str, Any]:
    obs_list = batch["obs"]
    action_masks = batch["action_masks"]
    node_masks = batch["node_masks"]
    actions = batch["actions"]

    old_logprobs = batch["old_logprobs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    indices = np.arange(len(obs_list))
    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clipfrac": [],
        "grad_norm": [],
    }

    stopped_early = False
    epochs_completed = 0

    for epoch in range(update_epochs):
        np.random.shuffle(indices)
        epoch_kls: list[float] = []

        for start in range(0, len(obs_list), minibatch_size):
            minibatch_indices = indices[start : start + minibatch_size]
            new_logprobs = []
            entropies = []
            values = []

            for idx in minibatch_indices:
                obs_i = obs_list[idx].clone().to(device)
                logp_i, entropy_i, value_i = agent.evaluate_action(
                    obs_i,
                    actions[idx],
                    action_masks[idx],
                    node_masks[idx],
                )
                new_logprobs.append(logp_i)
                entropies.append(entropy_i)
                values.append(value_i)

            new_logprobs_t = torch.stack(new_logprobs)
            entropies_t = torch.stack(entropies)
            values_t = torch.stack(values)

            mb_old_logprobs = old_logprobs[minibatch_indices]
            mb_advantages = advantages[minibatch_indices]
            mb_returns = returns[minibatch_indices]

            logratio = new_logprobs_t - mb_old_logprobs
            ratio = torch.exp(logratio)

            pg_loss_1 = -mb_advantages * ratio
            pg_loss_2 = -mb_advantages * torch.clamp(
                ratio, 1 - clip_coef, 1 + clip_coef
            )
            policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
            value_loss = 0.5 * ((values_t - mb_returns) ** 2).mean()
            entropy_loss = entropies_t.mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()
                clipfrac = (torch.abs(ratio - 1.0) > clip_coef).float().mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy_loss.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clipfrac"].append(clipfrac)
            metrics["grad_norm"].append(float(grad_norm))
            epoch_kls.append(approx_kl)

        epochs_completed = epoch + 1
        if (
            target_kl is not None
            and epoch_kls
            and float(np.mean(epoch_kls)) > target_kl
        ):
            stopped_early = True
            break

    with torch.no_grad():
        predicted_values = []
        for obs in obs_list:
            obs_i = obs.clone().to(device)
            predicted_values.append(agent.get_value(agent.forward(obs_i)).item())

    predicted_values_np = np.asarray(predicted_values, dtype=np.float32)
    returns_np = returns.detach().cpu().numpy().astype(np.float32)
    value_var = np.var(returns_np)
    explained_variance = (
        float("nan")
        if value_var == 0
        else float(1.0 - np.var(returns_np - predicted_values_np) / value_var)
    )

    result = {name: float(np.mean(values)) for name, values in metrics.items()}
    result["explained_variance"] = explained_variance
    result["stopped_early"] = bool(stopped_early)
    result["epochs_completed"] = int(epochs_completed)
    return result


def linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_training_plot(
    train_log: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> None:
    if not train_log:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    updates = [row["update"] for row in train_log]
    training_rewards = [row["mean_episode_return"] for row in train_log]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(updates, training_rewards)
    ax.set_title("Training reward")
    ax.set_xlabel("Update")
    ax.set_ylabel("Reward")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: str | Path,
    agent: GraphPPOAgent,
    optimizer: Optional[optim.Optimizer],
    update: int,
    train_log: Sequence[dict[str, Any]],
    action_schema: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "train_log": list(train_log),
        "schema": copy.deepcopy(action_schema) if action_schema is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


def get_schema_from_checkpoint_payload(
    payload: dict[str, Any],
) -> Optional[dict[str, Any]]:
    schema = payload.get("schema")
    if schema is not None:
        return schema

    shared_catalog = payload.get("shared_catalog")
    action_dims = payload.get("action_dims")
    if shared_catalog is None or action_dims is None:
        return None

    schema = {
        "shared_catalog": shared_catalog,
        "action_dims": action_dims,
    }
    if payload.get("obs_dims") is not None:
        schema["obs_dims"] = payload["obs_dims"]
    return schema


def load_checkpoint(
    path: str | Path,
    agent: GraphPPOAgent,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    expected_action_schema: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    checkpoint_schema = get_schema_from_checkpoint_payload(payload)

    if expected_action_schema is not None and checkpoint_schema is not None:
        assert (
            checkpoint_schema["action_dims"] == expected_action_schema["action_dims"]
        ), (
            "Checkpoint action_dims do not match expected schema: "
            f"{checkpoint_schema['action_dims']} != {expected_action_schema['action_dims']}"
        )
        if (
            expected_action_schema.get("obs_dims") is not None
            and checkpoint_schema.get("obs_dims") is not None
        ):
            assert (
                checkpoint_schema["obs_dims"] == expected_action_schema["obs_dims"]
            ), (
                "Checkpoint obs_dims do not match expected schema: "
                f"{checkpoint_schema['obs_dims']} != {expected_action_schema['obs_dims']}"
            )

    if checkpoint_schema is not None:
        assert dict(agent.action_dim) == checkpoint_schema["action_dims"], (
            "Agent action_dims do not match checkpoint action_dims: "
            f"{dict(agent.action_dim)} != {checkpoint_schema['action_dims']}"
        )
        if checkpoint_schema.get("obs_dims") is not None:
            assert dict(agent.obs_dim) == checkpoint_schema["obs_dims"], (
                "Agent obs_dims do not match checkpoint obs_dims: "
                f"{dict(agent.obs_dim)} != {checkpoint_schema['obs_dims']}"
            )

    agent.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload


def infer_best_training_reward(
    train_log: Sequence[dict[str, Any]],
    resume_payload: Optional[dict[str, Any]] = None,
) -> float:
    if resume_payload is not None:
        extra = resume_payload.get("extra") or {}
        best_from_payload = extra.get("best_train_reward")
        if best_from_payload is not None and np.isfinite(best_from_payload):
            return float(best_from_payload)
        best_from_payload = extra.get("best_train_greedy_return")
        if best_from_payload is not None and np.isfinite(best_from_payload):
            return float(best_from_payload)

    finite_returns = [
        float(row.get("mean_episode_return", float("nan")))
        for row in train_log
        if np.isfinite(row.get("mean_episode_return", float("nan")))
    ]
    return max(finite_returns) if finite_returns else -float("inf")


def format_metric(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def train_single_scenario(
    agent: GraphPPOAgent,
    optimizer: optim.Optimizer,
    train_env: GraphWrapper,
    train_scenario_id: str,
    action_schema: dict[str, Any],
    device: torch.device,
    checkpoint_dir: str | Path,
    plot_path: str | Path,
    config: TrainingConfig,
    resume_payload: Optional[dict[str, Any]] = None,
    checkpoint_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"

    train_log = (
        list(resume_payload.get("train_log", [])) if resume_payload is not None else []
    )
    best_train_reward = infer_best_training_reward(train_log, resume_payload)
    start_update = (
        int(resume_payload["update"]) + 1 if resume_payload is not None else 0
    )
    checkpoint_metadata = checkpoint_metadata or {}

    if start_update >= config.total_updates:
        LOG.info(
            "Resume checkpoint already reached requested total_updates=%s at update=%s.",
            config.total_updates,
            start_update - 1,
        )
        save_training_plot(train_log, plot_path)
        save_checkpoint(
            final_checkpoint_path,
            agent,
            optimizer,
            int(resume_payload["update"]),
            train_log,
            action_schema=action_schema,
            extra={
                **checkpoint_metadata,
                "type": "final",
                "train_scenario_id": str(train_scenario_id),
                "best_train_reward": best_train_reward,
            },
        )
        return {
            "train_log": train_log,
            "best_checkpoint_path": best_checkpoint_path,
            "final_checkpoint_path": final_checkpoint_path,
        }

    for update in range(start_update, config.total_updates):
        progress = update / max(config.total_updates - 1, 1)
        current_ent_coef = linear_schedule(
            config.entropy_start, config.entropy_end, progress
        )
        current_lr = (
            linear_schedule(config.learning_rate, config.learning_rate_end, progress)
            if config.lr_anneal
            else config.learning_rate
        )
        set_optimizer_lr(optimizer, current_lr)

        batch = collect_rollout(
            agent=agent,
            env=train_env,
            steps_per_rollout=config.steps_per_rollout,
            device=device,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            scenario_id=str(train_scenario_id),
        )

        metrics = ppo_update(
            agent=agent,
            optimizer=optimizer,
            batch=batch,
            device=device,
            clip_coef=config.clip_coef,
            ent_coef=current_ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            update_epochs=config.update_epochs,
            minibatch_size=config.minibatch_size,
            normalize_adv=True,
            target_kl=config.target_kl,
        )

        mean_episode_return = (
            float(np.mean(batch["episode_returns"]))
            if batch["episode_returns"]
            else float("nan")
        )

        row = {
            "update": update,
            "mean_episode_return": mean_episode_return,
            "scenario_counts": batch["scenario_counts"],
            "entropy_coef": current_ent_coef,
            "learning_rate": current_lr,
            **metrics,
        }
        train_log.append(row)

        if np.isfinite(mean_episode_return) and mean_episode_return > best_train_reward:
            best_train_reward = mean_episode_return
            save_checkpoint(
                best_checkpoint_path,
                agent,
                optimizer,
                update,
                train_log,
                action_schema=action_schema,
                extra={
                    **checkpoint_metadata,
                    "type": "best_train_reward",
                    "train_scenario_id": str(train_scenario_id),
                    "best_train_reward": best_train_reward,
                },
            )

        if ((update + 1) % config.save_every == 0) or (
            update == config.total_updates - 1
        ):
            periodic_path = checkpoint_dir / f"checkpoint_update_{update:04d}.pt"
            save_checkpoint(
                periodic_path,
                agent,
                optimizer,
                update,
                train_log,
                action_schema=action_schema,
                extra={
                    **checkpoint_metadata,
                    "type": "periodic",
                    "train_scenario_id": str(train_scenario_id),
                    "best_train_reward": best_train_reward,
                },
            )

        save_training_plot(train_log, plot_path)

        if (update % config.log_every == 0) or (update == config.total_updates - 1):
            LOG.info(
                "Update %s/%s | train_reward=%s | policy_loss=%s | value_loss=%s | entropy=%s",
                update + 1,
                config.total_updates,
                format_metric(mean_episode_return),
                format_metric(row["policy_loss"]),
                format_metric(row["value_loss"]),
                format_metric(row["entropy"]),
            )

    save_checkpoint(
        final_checkpoint_path,
        agent,
        optimizer,
        config.total_updates - 1,
        train_log,
        action_schema=action_schema,
        extra={
            **checkpoint_metadata,
            "type": "final",
            "train_scenario_id": str(train_scenario_id),
            "best_train_reward": best_train_reward,
        },
    )
    save_training_plot(train_log, plot_path)

    return {
        "train_log": train_log,
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
    }


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{device_arg}' but CUDA is not available."
        )
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single-scenario PrimAITE GNN PPO policy.",
    )
    parser.add_argument("--train-scenario", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--total-updates", type=int, default=DEFAULT_TRAINING_CONFIG.total_updates
    )
    parser.add_argument(
        "--no-packaged-schema",
        action="store_true",
        help="Do not include packaged PrimAITE configs when building the shared schema.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    training_config = TrainingConfig(total_updates=args.total_updates)
    device = resolve_device(args.device)
    set_seed(args.seed)

    train_scenario_path = (
        resolve_yaml_path(args.train_scenario)
        if args.train_scenario is not None
        else resolve_yaml_path(DEFAULT_TRAIN_SCENARIO)
    )

    LOG.info("Using device: %s", device)
    LOG.info("Training scenario: %s", train_scenario_path)
    LOG.info("Checkpoint directory: %s", args.checkpoint_dir)
    LOG.info("Training plot: %s", args.plot_path)

    train_cfg = load_cfg(train_scenario_path)
    action_schema = build_universal_schema(
        primary_cfgs=[train_cfg],
        include_packaged=not args.no_packaged_schema,
    )

    LOG.info("Schema action dims: %s", action_schema["action_dims"])
    LOG.info("Schema observation dims: %s", action_schema["obs_dims"])

    train_env: Optional[GraphWrapper] = None

    try:
        train_env = make_graph_env_from_cfg(train_cfg, action_schema)

        agent = build_agent_from_schema(
            action_schema=action_schema,
            hidden_dim=training_config.hidden_dim,
            device=device,
        )
        optimizer = optim.Adam(agent.parameters(), lr=training_config.learning_rate)

        resume_payload = None
        if args.resume is not None:
            resume_payload = load_checkpoint(
                args.resume,
                agent=agent,
                optimizer=optimizer,
                map_location=device,
                expected_action_schema=action_schema,
            )
            LOG.info(
                "Resumed from %s at update %s.", args.resume, resume_payload["update"]
            )

        checkpoint_metadata = {
            "train_scenario_path": str(train_scenario_path),
            "training_config": asdict(training_config),
            "device": str(device),
            "seed": args.seed,
        }

        result = train_single_scenario(
            agent=agent,
            optimizer=optimizer,
            train_env=train_env,
            train_scenario_id=train_scenario_path.name,
            action_schema=action_schema,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            plot_path=args.plot_path,
            config=training_config,
            resume_payload=resume_payload,
            checkpoint_metadata=checkpoint_metadata,
        )

        LOG.info("Best checkpoint: %s", result["best_checkpoint_path"])
        LOG.info("Final checkpoint: %s", result["final_checkpoint_path"])
        LOG.info("Training plot saved to: %s", args.plot_path)
    finally:
        if train_env is not None:
            train_env.close()


if __name__ == "__main__":
    main()
