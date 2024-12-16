import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# MLP Train
gym.register(
    id="Isaac-Gbagc-Cabinet_Opening-Direct-v0",
    entry_point=f"{__name__}.gbagc_cabinet_opening_same_freq_env:GbagcCabinetOpeningEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gbagc_cabinet_opening_same_freq_env:GbagcCabinetOpeningEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
