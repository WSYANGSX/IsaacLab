from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def task_complete(
    env: ManagerBasedRLEnv,
    pos_threshold: float,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Terminate when the cube pose are inside the threshold."""
    cube: RigidObject = env.scene[asset_cfg1.name]
    plate: RigidObject = env.scene[asset_cfg2.name]

    cube_pos_l = cube.data.root_pos_w - env.scene.env_origins
    plate_pos_l = plate.data.root_pos_w - env.scene.env_origins

    # calculate distance
    dist = torch.norm(cube_pos_l - plate_pos_l, p=2, dim=-1)
    task_complete = dist <= pos_threshold

    return task_complete
