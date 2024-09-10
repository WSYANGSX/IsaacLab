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
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Terminate when the cube pose are inside the threshold."""
    cube: RigidObject = env.scene[cube_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    cube_pos_l = cube.data.root_pos_w - env.scene.env_origins
    plate_pos_l = plate.data.root_pos_w - env.scene.env_origins

    # calculate distance
    dist = torch.norm(cube_pos_l - plate_pos_l, p=2, dim=-1)
    task_complete = dist <= pos_threshold

    return task_complete
