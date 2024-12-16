from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from my_projects.utils.math import rotation_distance

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def subgoal_reach(
    env: ManagerBasedRLEnv,
    pos_threshold: float,
    rot_threshold: float,
    subgoal_reach_bonus: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    subgoal_cmd_name: str = "subgoal",
) -> torch.Tensor:
    """Reward when subgoal achieved."""
    # extract the asset (to enable type hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_l = ee_frame.data.target_pos_source[:, 0]
    subgoal_pos_l = env.command_manager.get_command(subgoal_cmd_name)[:, :3]
    ee_subgoal_dist = torch.norm(ee_pos_l - subgoal_pos_l, p=2, dim=-1)

    ee_rot_l = ee_frame.data.target_quat_source[:, 0]
    subgoal_rot_l = env.command_manager.get_command(subgoal_cmd_name)[:, 3:7]
    ee_subgoal_rot_dist = rotation_distance(ee_rot_l, subgoal_rot_l)

    succ = (ee_subgoal_dist <= pos_threshold) & (ee_subgoal_rot_dist <= rot_threshold)

    reward = torch.where(
        succ,
        torch.full_like(succ, fill_value=subgoal_reach_bonus),
        torch.zeros_like(succ),
    )

    return reward


def task_goal_reach(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.39,
    final_goal_reach_bonus: float = 100,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg(
        "cabinet", joint_names=["drawer_top_joint"]
    ),
) -> torch.Tensor:
    """Reward when final goal achieved."""
    # extract the asset (to enable type hinting)
    cabinet: Articulation = env.scene[cabinet_cfg.name]

    top_drawer_joint_pos = torch.reshape(
        cabinet.data.joint_pos[:, cabinet_cfg.joint_ids], (env.num_envs,)
    )

    succ = top_drawer_joint_pos >= pos_threshold

    reward = torch.where(
        succ,
        torch.full_like(succ, fill_value=final_goal_reach_bonus),
        torch.zeros_like(succ),
    )
    return reward


def eposide_length(env: ManagerBasedRLEnv) -> torch.Tensor:
    reward = env.episode_length_buf

    return reward
