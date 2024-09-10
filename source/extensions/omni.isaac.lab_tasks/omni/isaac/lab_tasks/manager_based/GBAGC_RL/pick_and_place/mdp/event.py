from __future__ import annotations

import torch

from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import RigidObject, Articulation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def reset_asset_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # obtain the current pose
    default_pose_l = asset.data.default_root_state[:, :7]
    default_pos_w = default_pose_l[:, :3] + env.scene.env_origins
    default_rot_w = default_pose_l[:, 3:7]

    asset.write_root_pose_to_sim(torch.cat((default_pos_w, default_rot_w), dim=-1))
