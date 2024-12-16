from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from my_projects.utils.math import rotation_distance

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def get_handle_local_pose(
    env: ManagerBasedRLEnv,
    handle_frame_cfg: SceneEntityCfg = SceneEntityCfg("handle_frame"),
) -> torch.Tensor:
    """get local pose of end_effector"""
    # extract the asset (to enable type hinting)
    handle_frame: FrameTransformer = env.scene[handle_frame_cfg.name]
    # obtain panda hand pose
    handle_pos_l, handle_quat_l = (
        handle_frame.data.target_pos_source[..., 0, :],
        handle_frame.data.target_quat_source[..., 0, :],
    )

    return torch.cat((handle_pos_l, handle_quat_l), dim=-1)


def get_handle_local_pos(
    env: ManagerBasedRLEnv,
    handle_frame_cfg: SceneEntityCfg = SceneEntityCfg("handle_frame"),
) -> torch.Tensor:
    handle_frame: FrameTransformer = env.scene[handle_frame_cfg.name]
    # obtain panda hand pose
    handle_pos_l = handle_frame.data.target_pos_source[..., 0, :]

    return handle_pos_l


def get_handle_local_quat(
    env: ManagerBasedRLEnv,
    handle_frame_cfg: SceneEntityCfg = SceneEntityCfg("handle_frame"),
) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    handle_frame: FrameTransformer = env.scene[handle_frame_cfg.name]
    # obtain panda hand pose
    handle_quat_l = handle_frame.data.target_quat_source[..., 0, :]

    return handle_quat_l


def get_ee_local_pose(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """get local pose of end_effector"""
    # extract the asset (to enable type hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # obtain panda hand pose
    ee_pos_l, ee_quat_l = (
        ee_frame.data.target_pos_source[..., 0, :],
        ee_frame.data.target_quat_source[..., 0, :],
    )

    return torch.cat((ee_pos_l, ee_quat_l), dim=-1)


def get_ee_local_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # obtain panda hand pose
    ee_pos_l = ee_frame.data.target_pos_source[..., 0, :]

    return ee_pos_l


def get_ee_local_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # obtain panda hand pose
    ee_quat_l = ee_frame.data.target_quat_source[..., 0, :]

    return ee_quat_l


def get_ee_handle_dist(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    handle_frame_cfg: SceneEntityCfg = SceneEntityCfg("handle_frame"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_l = ee_frame.data.target_pos_source[..., 0, :]

    handle_frame: FrameTransformer = env.scene[handle_frame_cfg.name]
    handle_pos_l = handle_frame.data.target_pos_source[..., 0, :]

    return torch.norm(ee_pos_l - handle_pos_l, p=2, dim=-1, keepdim=True)


def get_ee_subgoal_dist(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    subgoal_cmd_name: str = "subgoals",
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_l = ee_frame.data.target_pos_source[..., 0, :]
    subgoal_pos_l = env.command_manager.get_command(subgoal_cmd_name)[:, :3]

    return torch.norm(ee_pos_l - subgoal_pos_l, p=2, dim=-1, keepdim=True)


def get_ee_subgoal_rot_dist(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    subgoal_cmd_name: str = "subgoals",
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_rot_l = ee_frame.data.target_quat_source[..., 0, :]
    subgoal_rot_l = env.command_manager.get_command(subgoal_cmd_name)[:, 3:7]

    return rotation_distance(ee_rot_l, subgoal_rot_l).unsqueeze(0).view(-1, 1)


def get_arm_dof_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
):
    """get the scaled joint position of robot arm."""
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    pos_data = asset.data.joint_pos[:, asset_cfg.joint_ids]

    return pos_data


def get_arm_dof_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
) -> torch.Tensor:
    """get the joint velocity of robot arm."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_data = asset.data.joint_vel[:, asset_cfg.joint_ids]

    return vel_data


def get_gripper_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["panda_finger_joint.*"]
    ),
) -> torch.Tensor:
    """get the joint position of robot gripper."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_data = asset.data.joint_pos[:, asset_cfg.joint_ids]

    return pos_data


def get_gripper_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["panda_finger_joint.*"]
    ),
) -> torch.Tensor:
    """get the joint velocity of robot gripper."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_data = asset.data.joint_vel[:, asset_cfg.joint_ids]

    return vel_data


def get_drawer_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "cabinet", joint_names=["drawer_top_joint"]
    ),
) -> torch.Tensor:
    """get the joint position of top drawer."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_data = asset.data.joint_pos[:, asset_cfg.joint_ids]

    return pos_data
