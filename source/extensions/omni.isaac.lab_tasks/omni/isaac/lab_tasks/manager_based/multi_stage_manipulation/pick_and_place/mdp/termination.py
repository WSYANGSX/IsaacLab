from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_mul, quat_conjugate

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def task_complete(
    env: ManagerBasedRLEnv,
    pos_threshold: float,
    rot_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Terminate when the cube pose are inside the threshold."""
    cube: RigidObject = env.scene[asset_cfg.name]

    # extract cube pos and rot
    cube_pos, cube_rot = cube.data.root_pos_w, cube.data.root_quat_w
    # extract goal pos and rot
    goal_pos, goal_rot = (
        env.goal_pos,
        env.goal_rot,
    )  # goal_pos and goal_rot在command中实现

    # calculate distance
    dist = torch.norm(cube_pos - goal_pos, p=2, dim=-1)
    rot_dist = rotation_distance(cube_rot, goal_rot)
    dist_complete = dist <= pos_threshold
    rot_complete = rot_dist <= rot_threshold
    task_complete = dist_complete.to(torch.int) & rot_complete.to(torch.int)

    return task_complete


""" Helper function """


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    )  # changed quat convention
