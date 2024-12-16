# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def lid_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_dist: float,
    lid_cfg: SceneEntityCfg = SceneEntityCfg("lid"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lid_frame_cfg: SceneEntityCfg = SceneEntityCfg("lid_frame"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    lid: RigidObject = env.scene[lid_cfg.name]
    lid_frame: FrameTransformer = env.scene[lid_frame_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Target object position: (num_envs, 3)
    lid_w = lid_frame.data.target_pos_w[..., 0, :]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    lid_ee_distance = torch.norm(lid_w - ee_w, dim=1)

    return (1 - torch.tanh(lid_ee_distance / std)) * ~(
        torch.norm(lid.data.root_pos_w[:, :2] - object.data.root_pos_w[:, :2])
        > minimal_dist
    )


def lid_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    lid_cfg: SceneEntityCfg = SceneEntityCfg("lid"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    lid: RigidObject = env.scene[lid_cfg.name]
    return torch.where(lid.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def lid_is_moved(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    minimal_dist: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lid_cfg: SceneEntityCfg = SceneEntityCfg("lid"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    lid: RigidObject = env.scene[lid_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    return torch.where(
        torch.norm(lid.data.root_pos_w[:, :2] - object.data.root_pos_w[:, :2])
        > minimal_dist,
        1.0,
        0.0,
    ) * torch.where(lid.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_dist: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lid_cfg: SceneEntityCfg = SceneEntityCfg("lid"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    lid: RigidObject = env.scene[lid_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return (1 - torch.tanh(object_ee_distance / std)) * (
        torch.norm(lid.data.root_pos_w[:, :2] - object.data.root_pos_w[:, :2])
        > minimal_dist
    )


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    # compute the desired position in the world frame
    des_pos_w = plate.data.root_pos_w[:, :3]
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (
        1 - torch.tanh(distance / std)
    )


def task_complete_bonus(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.01,
    final_goal_reach_bonus: float = 500,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Reward when task completed."""
    # extract the asset (to enable type hinting)
    object: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    cube_pos_l = object.data.root_pos_w - env.scene.env_origins
    plate_pos_l = plate.data.root_pos_w - env.scene.env_origins

    dist = torch.norm(cube_pos_l - plate_pos_l, p=2, dim=-1)
    succ = dist <= pos_threshold

    reward = torch.where(
        succ,
        torch.full_like(succ, fill_value=final_goal_reach_bonus),
        torch.zeros_like(succ),
    )
    return reward
