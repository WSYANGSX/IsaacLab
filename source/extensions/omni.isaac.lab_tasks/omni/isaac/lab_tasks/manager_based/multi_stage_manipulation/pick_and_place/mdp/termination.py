from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from my_project.rl_projects.utils.math import rotation_distance

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def task_complete(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.1,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Terminate when the cube pose are inside the threshold."""
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

    # calculate distance
    dist = torch.norm(cube_pos_l - target_pos_l, p=2, dim=-1)
    rot_dist = rotation_distance(cube_quat_l, target_quat_l)
    dist_complete = dist <= pos_threshold
    rot_complete = rot_dist <= rot_threshold
    task_complete = dist_complete.to(torch.int) & rot_complete.to(torch.int)

    return task_complete
