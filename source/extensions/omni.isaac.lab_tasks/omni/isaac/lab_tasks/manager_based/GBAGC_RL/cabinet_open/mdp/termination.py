from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def task_complete(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.39,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg(
        "cabinet", joint_names=["drawer_top_joint"]
    ),
) -> torch.Tensor:
    """Terminate when the cube pose are inside the threshold."""
    cabinet: Articulation = env.scene[cabinet_cfg.name]

    top_drawer_joint_pos = torch.reshape(
        cabinet.data.joint_pos[:, cabinet_cfg.joint_ids], (env.num_envs,)
    )

    task_complete = top_drawer_joint_pos >= pos_threshold

    return task_complete
