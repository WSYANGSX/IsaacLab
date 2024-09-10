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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
):
    """Reward when subgoal achieved."""
    # extract the asset (to enable type hinting)

    ee_subgoal_dist = get_ee_subgoal_dist(env, asset_cfg).view(
        -1,
    )
    ee_subgoal_rot_dist = get_ee_subgoal_rot_dist(env, asset_cfg).view(
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
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("plate"),
):
    """Reward when final goal achieved."""
    # extract the asset (to enable type hinting)
    asset1: RigidObject = env.scene[asset_cfg1.name]
    asset2: RigidObject = env.scene[asset_cfg2.name]

    asset1_pos_l = asset1.data.root_pos_w - env.scene.env_origins
    asset2_pos_l = asset2.data.root_pos_w - env.scene.env_origins

    dist = torch.norm(asset1_pos_l - asset2_pos_l, p=2, dim=-1)
    succ = dist <= pos_threshold

    reward = torch.where(
        succ,
        torch.full_like(succ, fill_value=final_goal_reach_bonus),
        torch.zeros_like(succ),
    )
    return reward
