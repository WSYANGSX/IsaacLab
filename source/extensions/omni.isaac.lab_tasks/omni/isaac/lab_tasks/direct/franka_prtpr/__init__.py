# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Prtpr environment.
"""

import gymnasium as gym

from . import agents
from .franka_prtpr_env import FrankaPrtprEnv, FrankaPrtprEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Cabinet-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_prtpr:FrankaPrtprEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPrtprEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
