import gymnasium as gym

from . import agents
from .prtpr_env_cfg import PrtprEnvCfg
from .prtpr_env import PrtprEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-PRTPR-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.prtpr:PrtprEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PrtprEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-PRTPR-Direct-Lstm-v0",
    entry_point="omni.isaac.lab_tasks.direct.prtpr:PrtprEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PrtprEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)