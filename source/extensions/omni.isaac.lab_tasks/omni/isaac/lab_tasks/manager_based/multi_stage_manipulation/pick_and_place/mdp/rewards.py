from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms
from my_project.rl_projects.utils.math import rotation_distance

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
    curr_subgoal = env.command_manager.get_command("subgoal_command")  # type: ignore
    curr_subgoal_pos = curr_subgoal[:, :3]
    curr_subgoal_quat = curr_subgoal[:, 3:7]

    # extract the current pose of end_effector gripper point
    # obtain panda hand pose
    panda_hand_pos_l = asset.data.root_pos_w - env.scene.env_origins
    panda_hand_quat_l = asset.data.root_quat_w

    # compute gripper frame local pose
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
    dist_succ = dist <= pos_threshold
    rot_succ = rot_dist <= rot_threshold
    subgoal_reach = dist_succ & rot_succ

    reward = torch.where(
        subgoal_reach == 1,
        torch.full_like(subgoal_reach, fill_value=subgoal_reach_bonus),
        torch.zeros_like(subgoal_reach),
    )
    return reward


def final_goal_reach(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.1,
    final_goal_reach_bonus: float = 100,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("target"),
):
    """Reward when final goal achieved."""
    # extract the asset (to enable type hinting)
    asset1: RigidObject = env.scene[asset_cfg1.name]
    asset2: RigidObject = env.scene[asset_cfg2.name]

    # extract the final goal
    target_pos_l, target_quat_l = (
        asset2.data.root_pos_w - env.scene.env_origins,
        asset2.data.root_quat_w,
    )

    # extract the current pose of cube
    cube_pos_l = asset1.data.root_pos_w - env.scene.env_origins
    cube_quat_l = asset1.data.root_quat_w

    # compute dist and rot dist
    # calculate distance
    dist = torch.norm(target_pos_l - cube_pos_l, p=2, dim=-1)
    rot_dist = rotation_distance(cube_quat_l, target_quat_l)
    dist_complete = dist <= pos_threshold
    rot_complete = rot_dist <= rot_threshold
    subgoal_reach = dist_complete & rot_complete

    reward = torch.where(
        subgoal_reach == 1,
        torch.full_like(subgoal_reach, fill_value=final_goal_reach_bonus),
        torch.zeros_like(subgoal_reach),
    )
    return reward
