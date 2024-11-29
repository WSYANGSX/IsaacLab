# command
# used for subgoal planner

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.managers import CommandTerm, CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils.configclass import configclass
from my_projects.utils.math import rotation_distance
from omni.isaac.lab.envs.mdp import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class SubgoalsCommand(CommandTerm):
    """Command generator that generates subgoals."""

    cfg: SubgoalsCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: SubgoalsCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)  # type: ignore

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_frame: FrameTransformer = env.scene["ee_frame"]

        self.robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        self.robot_entity_cfg.resolve(env.scene)
        self.hand_link_idx = self.robot_entity_cfg.body_ids[0]  # type: ignore

        self.pos_threshold = cfg.pos_threshold
        self.rot_threshold = cfg.rot_threshold
        
        
        # -- subgoals
        self.subgoals = (
            torch.tensor(cfg.subgoals_list, device=env.device)
            .unsqueeze(0)
            .repeat(env.num_envs, 1, 1)
        )

        self.subgoals_num = len(cfg.subgoals_list)

        print("Subgoals initialized. Subgoals num: ", self.subgoals_num)

        # --subgoal indices
        self.subgoals_indices = torch.zeros(
            env.num_envs, dtype=torch.int, device=env.device
        )

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.ee_pos_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_rot_command = torch.zeros(self.num_envs, 4, device=self.device)

        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_rot"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 3D-pose in base frame. Shape is (num_envs, 7)."""
        return torch.cat([self.ee_pos_command, self.ee_rot_command], dim=1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        ee_pos, ee_rot = (
            self.ee_frame.data.target_pos_source[:, 0, :],
            self.ee_frame.data.target_quat_source[:, 0, :],
        )

        self.metrics["error_pos"] = torch.norm(
            self.ee_pos_command - ee_pos,
            dim=-1,
        )
        self.metrics["error_rot"] = rotation_distance(ee_rot, self.ee_rot_command)

        # determine wherther to reach
        pos_reach = self.metrics["error_pos"] <= self.pos_threshold
        rot_reach = self.metrics["error_rot"] <= self.rot_threshold

        reach = pos_reach & rot_reach

        self.subgoals_indices = torch.where(
            reach, self.subgoals_indices + 1, self.subgoals_indices
        ).clamp(max=self.subgoals_num - 1)

    def _resample_command(self, env_ids: Sequence[int]):
        for env in env_ids:
            subgoals_for_env = self.subgoals[env]
            index_for_env = self.subgoals_indices[env]

            if index_for_env >= len(subgoals_for_env):
                raise IndexError(
                    f"Index {index_for_env} out of range for environment {env}"
                )

            subgoal = subgoals_for_env[index_for_env]
            position = subgoal[:3]
            rotation = subgoal[3:7]

            self.ee_pos_command[env] = position
            self.ee_rot_command[env] = rotation

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "subgoal_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()  # type:ignore
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/subgoal_pose"
                self.subgoal_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.subgoal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "subgoal_visualizer"):
                self.subgoal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.subgoal_visualizer.visualize(
            translations=self.ee_pos_command + self._env.scene.env_origins,
            orientations=self.ee_rot_command,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # resolve the environment IDs
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)  # type:ignore
        # set the command counter to zero
        self.command_counter[env_ids] = 0
        self.subgoals_indices[env_ids] = 0

        # resample the command
        self._resample(env_ids)  # type:ignore
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0
        return extras


@configclass
class SubgoalsCommandCfg(CommandTermCfg):
    """Configuration for the subgoals command generator."""

    class_type: type[SubgoalsCommand] = SubgoalsCommand

    asset_name: str = MISSING  # type:ignore
    """Name of the asset in the environment for which the commands are generated."""

    subgoals_list: list[list[float]] = MISSING  # type:ignore

    pos_threshold: float = MISSING

    rot_threshold: float = MISSING
