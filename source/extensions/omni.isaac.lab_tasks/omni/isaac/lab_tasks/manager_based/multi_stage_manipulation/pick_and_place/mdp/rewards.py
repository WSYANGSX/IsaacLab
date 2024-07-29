from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_mul, quat_conjugate

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def subgoal_reach(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.2,
    subgoal_reach_bonus: float = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
):
    """Reward when subgoal achieved."""
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # extract the subgoal of env
    curr_subgoal = env.subgoal  # type: ignore
    curr_subgoal_pos = curr_subgoal[:, :3]
    curr_subgoal_quat = curr_subgoal[:, 3:7]

    # extract the current pose of end_effector gripper point
    # obtain panda hand pose
    panda_hand_pos_l = asset.data.root_pos_w - env.scene.env_origins
    panda_hand_quat_l = asset.data.root_quat_w

    # compute gripper frame local pose
    gripper_frame_in_hand_frame = (0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0)
    gripper_frame_in_hand_frame = torch.tensor(
        (0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0), device=env.device
    ).repeat(env.scene.num_envs, 1)

    ee_local_pos, ee_local_quat = combine_frame_transforms(
        t01=panda_hand_pos_l,
        q01=panda_hand_quat_l,
        t12=gripper_frame_in_hand_frame[:, :3],
        q12=gripper_frame_in_hand_frame[:, 3:7],
    )

    # compute dist and rot dist
    # calculate distance
    dist = torch.norm(ee_local_pos - curr_subgoal_pos, p=2, dim=-1)
    rot_dist = rotation_distance(ee_local_quat, curr_subgoal_quat)
    dist_complete = dist <= pos_threshold
    rot_complete = rot_dist <= rot_threshold
    subgoal_reach = dist_complete.to(torch.int) & rot_complete.to(torch.int)

    reward = torch.where(
        subgoal_reach == 1,
        torch.full_like(subgoal_reach, fill_value=subgoal_reach_bonus),
        torch.zeros_like(subgoal_reach),
    )
    return reward


def final_goal_reach(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.1,
    rot_threshold: float = 0.2,
    final_goal_reach_bonus: float = 100,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
):
    """Reward when final goal achieved."""
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # extract the final goal
    goal_pos, goal_quat = (
        env.goal_pos,
        env.goal_quat,
    )  # 在command manager中设置VisualiztionMaker定义

    # extract the current pose of end_effector gripper point
    # obtain panda hand pose
    object_pos_l = asset.data.root_pos_w - env.scene.env_origins
    object_quat_l = asset.data.root_quat_w

    # compute dist and rot dist
    # calculate distance
    dist = torch.norm(object_pos_l - goal_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_quat_l, goal_quat)
    dist_complete = dist <= pos_threshold
    rot_complete = rot_dist <= rot_threshold
    subgoal_reach = dist_complete.to(torch.int) & rot_complete.to(torch.int)

    reward = torch.where(
        subgoal_reach == 1,
        torch.full_like(subgoal_reach, fill_value=final_goal_reach_bonus),
        torch.zeros_like(subgoal_reach),
    )
    return reward


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    )  # changed quat convention
