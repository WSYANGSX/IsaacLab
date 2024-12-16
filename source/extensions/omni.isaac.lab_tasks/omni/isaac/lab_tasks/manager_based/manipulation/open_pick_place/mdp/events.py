from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reset_lid_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    lid_cfg: SceneEntityCfg = SceneEntityCfg("lid"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    object: RigidObject = env.scene[object_cfg.name]
    lid: RigidObject = env.scene[lid_cfg.name]

    object_pos_w = object.data.root_pos_w[env_ids, :3]

    root_rot_w = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
    ).repeat(len(env_ids), 1)

    root_pose = torch.cat((object_pos_w, root_rot_w), dim=-1)

    lid.write_root_pose_to_sim(root_pose, env_ids)
