from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm, CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils.configclass import configclass
from my_project.rl_projects.utils.math import rotation_distance

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class SubgoalCommand(CommandTerm):
    """Command generator that generates subgoals."""

    cfg: SubgoalCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: SubgoalCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)  # type: ignore

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # obtain subgoals list
        self.subgoals = cfg.subgoals_list
        self.subgoals = torch.tensor(
            self.subgoals, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        self.subgoals_nums = self.subgoals.size()[-2]
        self.subgoals_counter = torch.zeros(self.num_envs, device=self.device)

        # crete buffers to store the command
        # -- commands: (pos, quat)
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_l = torch.zeros_like(self.pose_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_quat"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "SubgoalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 3D-pose in local frame. Shape is (num_envs, 7)."""
        return self.pose_command_l

    """
    Implementation specific functions.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """reset subgoals"""
        if env_ids is None:
            env_ids = slice(None)
        # set the command counter to zero
        self.command_counter[env_ids] = 0
        self.subgoals_counter[env_ids] = (
            0  # self._resample()应当以subgoals_counter为依据进行采样
        )
        # resample the command
        self._resample(env_ids)
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0
        return extras

    def _update_metrics(self):
        # logs data
        env_obs = self._env.observation_manager.compute_group("policy")
        ee_pose = env_obs[:, 7:14]  # type: ignore

        self.metrics["error_pos"] = torch.norm(
            self.pose_command_l[:, :3] - ee_pose[:, :3], dim=1
        )
        self.metrics["error_quat"] = rotation_distance(
            ee_pose[:3:7], self.pose_command_l[:, 3:7]
        )

        # determine whether the subgoal has changed
        pos_succ = torch.where(
            self.metrics["error_pos"] <= 0.05,
            torch.ones_like(self.metrics["error_pos"], device=self.device),
            torch.zeros_like(self.metrics["error_pos"], device=self.device),
        )
        rot_succ = torch.where(
            self.metrics["error_pos"] <= 0.2,
            torch.ones_like(self.metrics["error_quat"], device=self.device),
            torch.zeros_like(self.metrics["error_quat"], device=self.device),
        )
        succ = pos_succ & rot_succ
        self.subgoals_counter = torch.where(
            succ == 1,
            self.subgoals_counter + torch.ones_like(self.subgoals_counter),
            self.subgoals_counter,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        for i in range(len(env_ids)):
            self.pose_command_l[env_ids[i]] = self.subgoals[env_ids[i]][
                self.subgoals_counter[env_ids[i]]
            ]
        # offset the position command by the env origins
        self.pose_command_w[:] = self.pose_command_l
        self.pose_command_w[:, :3] += (
            self._env.scene.env_origins
        )  # for visualization makers

    def _update_command(self):
        """Re-target the position command to the current root state."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "subgoal_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/subgoal_goal"
                self.subgoal_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.subgoal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "arrow_goal_visualizer"):
                self.subgoal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.subgoal_visualizer.visualize(
            translations=self.pose_command_w[:, :3],
            orientations=self.pose_command_w[:, 3:7],
        )


@configclass
class SubgoalCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = SubgoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    subgoals_list: list = MISSING
    """List of subgoals in the environment for the robot to achieve sequentially."""
