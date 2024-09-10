from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from .observations import (
    get_ee_subgoal_dist,
    get_ee_subgoal_rot_dist,
)

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def subgoal_reach(
    env: ManagerBasedRLEnv,
    pos_threshold: float,
    rot_threshold: float,
    subgoal_reach_bonus: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Reward when subgoal achieved."""
    # extract the asset (to enable type hinting)

    ee_subgoal_dist = get_ee_subgoal_dist(env, ee_frame_cfg).view(
        -1,
    )
    ee_subgoal_rot_dist = get_ee_subgoal_rot_dist(env, ee_frame_cfg).view(
        -1,
    )

    succ = (ee_subgoal_dist <= pos_threshold) & (ee_subgoal_rot_dist <= rot_threshold)

    reward = torch.where(
        succ,
        torch.full_like(succ, fill_value=subgoal_reach_bonus),
        torch.zeros_like(succ),
    )

    return reward


def task_goal_reach(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.02,
    final_goal_reach_bonus: float = 100,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
):
    """Reward when final goal achieved."""
    # extract the asset (to enable type hinting)
    cube: RigidObject = env.scene[cube_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    cube_pos_l = cube.data.root_pos_w - env.scene.env_origins
    plate_pos_l = plate.data.root_pos_w - env.scene.env_origins

    dist = torch.norm(cube_pos_l - plate_pos_l, p=2, dim=-1)
    succ = dist <= pos_threshold

    reward = torch.where(
        succ,
        torch.full_like(succ, fill_value=final_goal_reach_bonus),
        torch.zeros_like(succ),
    )
    return reward
