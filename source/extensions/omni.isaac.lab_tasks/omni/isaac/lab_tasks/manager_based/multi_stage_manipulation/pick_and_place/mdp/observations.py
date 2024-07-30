from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs.mdp import joint_pos, joint_vel
from omni.isaac.lab.utils.math import (
    combine_frame_transforms,
    quat_mul,
    quat_conjugate,
    quat_unique,
)

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def get_asset_local_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Asset root position in the local frame."""

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # obtain the current pose
    curr_pos_w = asset.data.root_pos_w
    curr_pos_l = curr_pos_w - env.scene.env_origins

    return curr_pos_l


def get_asset_local_rot(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, make_quat_unique: bool = False
) -> torch.Tensor:
    """Asset root position in the local frame."""

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # obtain the current pose
    curr_quat_w = asset.data.root_quat_w
    curr_quat_l = curr_quat_w

    return quat_unique(curr_quat_l) if make_quat_unique else curr_quat_l


def get_grip_point_local_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """get local position and oritation of end_effector"""

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # obtain panda hand pose
    panda_hand_pos_l = (
        asset.data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins
    )
    panda_hand_quat_l = asset.data.body_quat_w[:, asset_cfg.body_ids]

    # compute gripper frame local pose
    gripper_frame_in_hand_frame = (0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0)
    gripper_frame_in_hand_frame = torch.tensor(
        gripper_frame_in_hand_frame, device=env.device
    ).repeat(env.scene.num_envs, 1)

    grip_point_pos_l, grip_point_quat_l = combine_frame_transforms(
        t01=panda_hand_pos_l,
        q01=panda_hand_quat_l,
        t12=gripper_frame_in_hand_frame[:, :3],
        q12=gripper_frame_in_hand_frame[:, 3:7],
    )

    return torch.cat((grip_point_pos_l, grip_point_quat_l), dim=-1)


def get_subgoal_local_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Asset target position and orentation in the local frame."""
    # extract goal pose
    goal_pos, goal_quat = env.command_manager.get_command("target_pose")

    return torch.cat((goal_pos, goal_quat), dim=-1)


def get_grip_goal_dist(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    grip_point_pose = get_grip_point_local_pose(env, asset_cfg)
    subgoal_pose = get_subgoal_local_pose(env)

    grip_pos = grip_point_pose[:, :3]
    subgoal_pos = subgoal_pose[:, :3]

    return torch.norm(grip_pos - subgoal_pos, p=2, dim=-1)


def get_grip_goal_rot_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    grip_point_pose = get_grip_point_local_pose(env, asset_cfg)
    subgoal_pose = get_subgoal_local_pose(env)

    grip_rot = grip_point_pose[:, 3:7]
    subgoal_rot = subgoal_pose[:, 3:7]

    return rotation_distance(subgoal_rot, grip_rot)


def get_arm_position(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """get the joint position of robot arm."""

    asset_cfg.joint_names = ["panda_joint.*"]
    pos_data = joint_pos(env, asset_cfg)

    return pos_data


def get_arm_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """get the joint velocity of robot arm."""

    asset_cfg.joint_names = ["panda_joint.*"]
    vel_data = joint_vel(env, asset_cfg)

    return vel_data


def get_gripper_position(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """get the joint position of robot gripper."""

    asset_cfg.joint_names = ["panda_finger_joint.*"]
    pos_data = joint_pos(env, asset_cfg)

    return pos_data


def get_gripper_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """get the joint velocity of robot gripper."""

    asset_cfg.joint_names = ["panda_finger_joint.*"]
    vel_data = joint_vel(env, asset_cfg)

    return vel_data


"""
Helper function
"""


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    )  # changed quat convention
