# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Prtpr environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# MLP Train
gym.register(
    id="Isaac-Franka_Prtpr-Direct-ManipulationSphere-v0",
    entry_point=f"{__name__}.manipulation_sphere_env:ManipulationSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manipulation_sphere_env:ManipulationSphereEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka_Prtpr-Direct-JointSpace-v0",
    entry_point=f"{__name__}.joint_space_env:JointSpaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_space_env:JointSpaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_jointspace_cfg.yaml",
    },
)

# LSTM Train
gym.register(
    id="Isaac-Franka_Prtpr-Direct-ManipulationSphere-Lstm",
    entry_point=f"{__name__}.manipulation_sphere_env:ManipulationSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manipulation_sphere_env:ManipulationSphereEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)


# Play
gym.register(
    id="Isaac-Franka_Prtpr-Direct-ManipulationSphere-Play-v0",
    entry_point=f"{__name__}.manipulation_sphere_env:ManipulationSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manipulation_sphere_env:ManipulationSphereEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka_Prtpr-Direct-JointSpace-Play-v0",
    entry_point=f"{__name__}.joint_space_env:JointSpacePlayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_space_env:JointSpaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_jointspace_cfg.yaml",
    },
)
