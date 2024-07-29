from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs.mdp import joint_pos, joint_vel
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_mul, quat_conjugate

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def get_asset_local_pose(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """get position and oritation of cube

    The function computes the position and the orientation of the asset's body (in world frame).
    """

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # obtain the current pose
    curr_pos_w = asset.data.root_pos_w
    curr_quat_w = asset.data.root_quat_w
    curr_pos_l = curr_pos_w - env.scene.env_origins
    curr_quat_l = curr_quat_w

    return torch.cat((curr_pos_l, curr_quat_l), dim=-1)


def get_ee_local_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """get local position and oritation of end_effector"""

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # obtain panda hand pose
    panda_hand_pos_l = asset.data.root_pos_w - env.scene.env_origins
    panda_hand_quat_l = asset.data.root_quat_w

    # compute gripper frame local pose
    gripper_frame_in_hand_frame = (0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0)
    gripper_frame_in_hand_frame = torch.tensor(
        gripper_frame_in_hand_frame, device=env.device
    ).repeat(env.scene.num_envs, 1)

    ee_local_pos, ee_local_quat = combine_frame_transforms(
        t01=panda_hand_pos_l,
        q01=panda_hand_quat_l,
        t12=gripper_frame_in_hand_frame[:, :3],
        q12=gripper_frame_in_hand_frame[:, 3:7],
    )

    return torch.cat((ee_local_pos, ee_local_quat), dim=-1)


def get_goal_local_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """get position and oritation of cube

    The function computes the position and the orientation of the asset's body (in world frame).
    """

    # extract goal pose
    goal_pos, goal_quat = (
        env.goal_pos,
        env.goal_quat,
    )  # 在command manager中设置VisualiztionMaker定义

    return torch.cat((goal_pos, goal_quat), dim=-1)


def get_ee_goal_dist(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    ee_pose = get_ee_local_pose(env, asset_cfg)
    goal_pose = get_subgoal(env)

    ee_pos = ee_pose[:, :3]
    goal_pos = goal_pose[:, :3]

    return torch.norm(ee_pos - goal_pos, p=2, dim=-1)


def get_ee_goal_rot_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    ee_pose = get_ee_local_pose(env, asset_cfg)
    goal_pose = get_subgoal(env)

    ee_rot = ee_pose[:, 3:7]
    goal_rot = goal_pose[:, 3:7]

    return rotation_distance(ee_rot, goal_rot)


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


def get_subgoal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.subgoal  # type: ignore


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    )  # changed quat convention
