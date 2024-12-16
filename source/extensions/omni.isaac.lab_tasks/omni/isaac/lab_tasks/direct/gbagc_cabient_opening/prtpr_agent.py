from __future__ import annotations

import os
import yaml
import copy

import math
import torch
import numpy as np
import gymnasium as gym

from rl_games.algos_torch import model_builder
from rl_games.algos_torch import torch_ext
from rl_games.common.tr_helpers import unsqueeze_obs
from omni.isaac.lab_tasks.utils import parse_env_cfg


@torch.jit.script
def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class PrtprAgent(object):
    """Warpper class for rl_game BasePlayer."""

    def __init__(self, task_name: str, device: str | torch.device, checkpoint_path: str) -> None:
        self.task_name = task_name
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.policy_root_path = os.path.split(os.path.split(checkpoint_path)[0])[0]
        self.policy_root_path = os.path.abspath(self.policy_root_path)
        print(f"[INFO] Loading experiment from directory: {self.policy_root_path}")

        # 加载agent配置参数
        agent_cfg_path = os.path.join(self.policy_root_path, "params/agent.yaml")
        self.agent_cfg = self.load_agent_parameters(agent_cfg_path)

        self.agent_config = self.agent_cfg["params"]["config"]  # type: ignore
        self.normalize_input = self.agent_config.get("normalize_input", False)
        self.normalize_value = self.agent_config.get("normalize_value", False)
        self.clip_actions = self.agent_cfg["params"].get("env").get("clip_actions", math.inf)  # type: ignore
        self.clip_observations = self.agent_cfg["params"].get("env").get("clip_observations", math.inf)  # type: ignore
        self.device_name = self.agent_config["device_name"]
        self.device = torch.device(self.device_name)
        self.num_agents = self.agent_cfg["params"].get("env").get("agents", 1)  # type: ignore
        self.player_config = self.agent_config.get("player", {})
        self.has_central_value = self.agent_config.get("central_value_config") is not None

        # 加载环境配置参数
        self.env_cfg = parse_env_cfg(task_name, device=self.device)

        if self.env_cfg.action_space is not None:
            self.action_space = self.env_cfg.action_space
            self.actions_num = self.action_space.shape[0]
        elif self.env_cfg.num_actions:
            self.actions_num = self.env_cfg.num_actions
            self.action_space = gym.spaces.Box(
                np.ones(self.actions_num, dtype=np.float32) * -self.clip_actions,
                np.ones(self.actions_num, dtype=np.float32) * self.clip_actions,
            )

        if self.env_cfg.observation_space is not None:
            self.observation_space = self.env_cfg.observation_space
            self.observations_num = self.observation_space.shape[0]
        elif self.env_cfg.num_observations:
            self.observations_num = self.env_cfg.num_observations
            self.observation_space = gym.spaces.Box(
                np.ones(self.observations_num, dtype=np.float32) * -self.clip_observations,
                np.ones(self.observations_num, dtype=np.float32) * self.clip_observations,
            )

        # 配置参数
        self.states = None
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False

        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)

        self.obs_low = torch.from_numpy(self.observation_space.low.copy()).float().to(self.device)
        self.obs_high = torch.from_numpy(self.observation_space.high.copy()).float().to(self.device)

        self.obs_shape = self.observation_space.shape

        # loab networks
        self.load_networks(self.agent_cfg["params"])  # type: ignore

        obs_shape = self.obs_shape
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.agent_cfg["params"].get("env").get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        self.network = self.agent_config["network"]
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

        # find checkpoint
        self.restore(self.checkpoint_path)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.agent_config["network"] = builder.load(params)

    def load_env_parameters(self, file_path: str):
        # parse the default config file
        if isinstance(file_path, str) and file_path.endswith(".yaml"):
            config_file = file_path
            # load the configuration
            print(f"[INFO]: Parsing configuration from: {config_file}")
            with open(config_file, encoding="utf-8") as f:
                cfg = yaml.full_load(f)
            return cfg
        else:
            raise ValueError(f"{file_path} is not a valid yaml file path")

    def load_agent_parameters(self, file_path: str):
        # parse the default config file
        if isinstance(file_path, str) and file_path.endswith(".yaml"):
            config_file = file_path
            # load the configuration
            print(f"[INFO]: Parsing configuration from: {config_file}")
            with open(config_file, encoding="utf-8") as f:
                cfg = yaml.full_load(f)
            return cfg
        else:
            raise ValueError(f"{file_path} is not a valid yaml file path")

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def get_action(self, obs, is_deterministic=False):
        if not self.has_batch_dimension:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if not self.has_batch_dimension:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def restore(self, fn: str | None = None):
        if fn is not None:
            checkpoint = torch_ext.load_checkpoint(fn)
        else:
            checkpoint = torch_ext.load_checkpoint(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [
                torch.zeros((s.size()[0], self.batch_size, s.size()[2]), dtype=torch.float32).to(self.device)
                for s in rnn_states
            ]

    def reset(self):
        self.init_rnn()

    def get_weights(self):
        weights = {}
        weights["model"] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights["model"])
        if self.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def get_batch_size(self, obses, batch_size=1):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if "obs" in obses:
                obses = obses["obs"]
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if "observation" in obses:
                first_key = "observation"
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size


if __name__ == "__main__":
    prtpr = PrtprAgent(
        "Isaac-Franka_Prtpr-Direct-JointSpace-v0",
        "cuda:0",
        "/home/yangxf/my_projects/IsaacLab/logs/rl_games/franka_prtpr_jointspace_direct/v2/nn/last_franka_prtpr_jointspace_direct_ep_3800_rew_3609.105.pth",
    )
    prtpr.reset()

    obs = torch.randn((19, 19), device="cuda:0", dtype=torch.float32)
    # if prtpr.is_rnn:
    #     prtpr.init_rnn()
    batch_size = 1
    prtpr.get_batch_size(obses=obs, batch_size=batch_size)
    print(prtpr.batch_size)
    act = prtpr.get_action(obs)
    print(act)
