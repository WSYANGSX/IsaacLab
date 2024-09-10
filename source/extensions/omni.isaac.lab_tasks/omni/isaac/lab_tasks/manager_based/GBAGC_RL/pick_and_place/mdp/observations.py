from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import quat_mul, quat_conjugate
from omni.isaac.lab.envs.mdp.observations import generated_commands

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def get_asset_local_pose(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """get position and oritation of asset.

    The function computes the position and the orientation of the asset's body (in local frame).
    """

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # obtain the current pose
    curr_pos_w = asset.data.root_pos_w
    curr_quat_w = asset.data.root_quat_w
    curr_pos_l = curr_pos_w - env.scene.env_origins
    curr_quat_l = curr_quat_w

    return torch.cat((curr_pos_l, curr_quat_l), dim=-1)


def get_asset_local_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset_local_pose = get_asset_local_pose(env, asset_cfg)

    return asset_local_pose[:, :3]


def get_asset_local_quat(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset_local_pose = get_asset_local_pose(env, asset_cfg)

    return asset_local_pose[:, 3:7]


def get_ee_local_pose(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """get local pose of end_effector"""

    # extract the asset (to enable type hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # obtain panda hand pose
    ee_pos_l, ee_quat_l = (
        ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins,
        ee_frame.data.target_quat_w[:, 0, :],
    )

    return torch.cat((ee_pos_l, ee_quat_l), dim=-1)


def get_ee_local_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_local_pose = get_ee_local_pose(env, ee_frame_cfg)

    return ee_local_pose[:, :3]


def get_ee_local_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_local_pose = get_ee_local_pose(env, ee_frame_cfg)

    return ee_local_pose[:, 3:7]


def get_ee_cube_dist(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    ee_pos = get_ee_local_pos(env, ee_frame_cfg)
    cube_pos = get_asset_local_pos(env, cube_cfg)

    return torch.norm(ee_pos - cube_pos, p=2, dim=-1, keepdim=True)


def get_ee_cube_rot_dist(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    ee_rot = get_ee_local_quat(env, ee_frame_cfg)
    cube_rot = get_asset_local_quat(env, cube_cfg)

    return rotation_distance(ee_rot, cube_rot).unsqueeze(0).view(-1, 1)


def get_ee_subgoal_dist(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_pos = get_ee_local_pos(env, ee_frame_cfg)
    subgoal_pos = generated_commands(env, "subgoals")[:, :3]

    return torch.norm(ee_pos - subgoal_pos, p=2, dim=-1, keepdim=True)


def get_ee_subgoal_rot_dist(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_rot = get_ee_local_quat(env, ee_frame_cfg)
    subgoal_rot = generated_commands(env, "subgoals")[:, 3:7]

    return rotation_distance(ee_rot, subgoal_rot).unsqueeze(0).view(-1, 1)


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
