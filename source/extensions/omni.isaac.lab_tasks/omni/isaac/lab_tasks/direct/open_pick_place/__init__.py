import gymnasium as gym

from . import agents
from .open_pick_place_env import OpenPickPlaceEnv, OpenPickPlaceEnvCfg

##
# Register Gym environments.
##

# MLP Train
gym.register(
    id="Isaac-Open_Pick_Place-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.open_pick_place:OpenPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenPickPlaceEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
