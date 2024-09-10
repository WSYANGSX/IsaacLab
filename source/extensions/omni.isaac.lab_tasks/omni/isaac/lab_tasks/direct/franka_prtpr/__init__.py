# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Prtpr environment.
"""

import gymnasium as gym

from . import agents
from .manipulation_sphere_env import (
    ManipulationSphereEnv,
    ManipulationSpherePlayEnv,
    ManipulationSphereEnvCfg,
)
from .joint_space_env import JointSpaceEnv, JointSpacePlayEnv, JointSpaceEnvCfg
##
# Register Gym environments.
##

# MLP Train
gym.register(
    id="Isaac-FrankaPrtpr-Direct-ManipulationSphere-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_prtpr:ManipulationSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ManipulationSphereEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-FrankaPrtpr-Direct-JointSpace-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_prtpr:JointSpaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JointSpaceEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_jointspace_cfg.yaml",
    },
)

# LSTM Train
gym.register(
    id="Isaac-FrankaPrtpr-Direct-Lstm-ManipulationSphere",
    entry_point="omni.isaac.lab_tasks.direct.franka_prtpr:ManipulationSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ManipulationSphereEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)


# Play
gym.register(
    id="Isaac-FrankaPrtpr-Direct-ManipulationSphere-Play-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_prtpr:ManipulationSpherePlayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ManipulationSphereEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-FrankaPrtpr-Direct-JointSpace-Play-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_prtpr:JointSpacePlayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JointSpaceEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_jointspace_cfg.yaml",
    },
)
