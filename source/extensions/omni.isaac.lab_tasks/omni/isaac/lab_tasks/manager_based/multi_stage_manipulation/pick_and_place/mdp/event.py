from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import random_yaw_orientation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reset_cube_pose(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> None:
    """Reset cube pose by offset."""
    asset: RigidObject = env.scene[asset_cfg.name]

    default_pos = asset.data.default_root_state[env_ids][:, :3]

    default_rot = asset.data.default_root_state[env_ids][:, 3:7]

    asset.write_root_pose_to_sim(
        torch.cat((default_pos, default_rot), dim=-1), env_ids=env_ids
    )


def reset_cube_pose_offset(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> None:
    """Reset cube pose by offset."""
    asset: RigidObject = env.scene[asset_cfg.name]

    default_pos = asset.data.default_root_state[env_ids][:, :3]
    offset = torch.zeros_like(default_pos)
    offset[:, :2] = (torch.rand(len(env_ids), 2, device=env.device) * 2 - 1) * 0.15

    pos_offset = default_pos + offset + env.scene.env_origins[env_ids]
    random_rot = random_yaw_orientation(len(env_ids), device=env.device)

    asset.write_root_pose_to_sim(
        torch.cat((pos_offset, random_rot), dim=-1), env_ids=env_ids
    )
