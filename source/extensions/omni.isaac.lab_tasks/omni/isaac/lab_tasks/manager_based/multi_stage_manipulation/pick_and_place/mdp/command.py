from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm, CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils.configclass import configclass
from omni.isaac.lab.utils.math import (
    quat_from_euler_xyz,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)

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

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

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
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat(
            [self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1
        )

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos_2d"] = torch.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = (
                self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            )
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(
                target_direction - self.robot.data.heading_w[env_ids]
            ).abs()
            curr_to_flipped_target = wrap_to_pi(
                flipped_target_direction - self.robot.data.heading_w[env_ids]
            ).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(
            yaw_quat(self.robot.data.root_quat_w), target_vec
        )
        self.heading_command_b[:] = wrap_to_pi(
            self.heading_command_w - self.robot.data.heading_w
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "arrow_goal_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
                marker_cfg.prim_path = "/Visuals/Command/pose_goal"
                self.arrow_goal_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.arrow_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "arrow_goal_visualizer"):
                self.arrow_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.arrow_goal_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )


@configclass
class SubgoalCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = SubgoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""
