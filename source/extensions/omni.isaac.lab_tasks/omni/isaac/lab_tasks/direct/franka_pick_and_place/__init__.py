import gymnasium as gym

from . import agents
from .pick_and_place_env import OpenPickPlaceEnv, PickAndPlaceEnvCfg

##
# Register Gym environments.
##

# MLP Train
gym.register(
    id="Isaac-Franka-Pick_And_Place-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_pick_and_place:OpenPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenPickPlaceEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
