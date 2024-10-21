import gymnasium as gym

from . import agents
from .gbagc_franka_cabinet_env import GbagcFrankaCabinetEnv, GbagcFrankaCabinetEnvCfg

##
# Register Gym environments.
##

# MLP Train
gym.register(
    id="Isaac-Gbagc-CabinetOpening-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.gbagc_franka_cabient:GbagcFrankaCabinetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GbagcFrankaCabinetEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
