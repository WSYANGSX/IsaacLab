import math
import os
import torch

from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg


class PrtprAgent:
    def __init__(
        self,
        env: DirectRLEnv | ManagerBasedRLEnv,
        task_name: str,
        device: str | torch.device,
        checkpoint: str,
    ) -> None:
        self.device = device
        self.env_cfg = parse_env_cfg(task_name, device=device)
        self.agent_cfg = load_cfg_from_registry(task_name, "rl_games_cfg_entry_point")
        self.log_root_path = os.path.join("logs", "rl_games", self.agent_cfg["params"]["config"]["name"])
        self.log_root_path = os.path.abspath(self.log_root_path)
        print(f"[INFO] Loading experiment from directory: {self.log_root_path}")

        resume_path = retrieve_file_path(checkpoint)
        self.log_dir = os.path.dirname(os.path.dirname(resume_path))

        # wrap around environment for rl-games
        self.rl_device = self.agent_cfg["params"]["config"]["device"]
        self.clip_obs = self.agent_cfg["params"]["env"].get("clip_observations", math.inf)
        self.clip_actions = self.agent_cfg["params"]["env"].get("clip_actions", math.inf)

        # load previously trained model
        self.agent_cfg["params"]["load_checkpoint"] = True
        self.agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {self.agent_cfg['params']['load_path']}")

        # set number of actors into agent config
        self.agent_cfg["params"]["config"]["num_actors"] = env.num_envs
        # create runner from rl-games
        self.runner = Runner()
        self.runner.load(self.agent_cfg)
        # obtain the agent from the runner
        self.agent: BasePlayer = self.runner.create_player()
        self.agent.restore(resume_path)
        self.agent.reset()
        self.has_batch_dimension = False

    def get_batch_size(self, obs):
        _ = self.agent.get_batch_size(obs, 1)
        self.has_batch_dimension = True

    def get_action(self, obs, is_deterministic=True):
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        actions = self.agent.get_action(obs, is_deterministic)
        return torch.clamp(actions, -self.clip_actions, self.clip_actions)
