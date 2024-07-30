from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs.mdp import SceneEntityCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.managers.observation_manager import (
    ObservationGroupCfg as ObsGroup,
    ObservationManager,
)
from omni.isaac.lab.managers.observation_manager import ObservationTermCfg as ObsTerm
from omni.isaac.lab.markers import FRAME_MARKER_CFG, VisualizationMarkers
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.math import (
    quat_from_angle_axis,
    quat_mul,
    quat_conjugate,
    normalize,
    combine_frame_transforms,
    unscale_transform,
)
from .observations import (
    get_grip_point_local_pose,
    get_subgoal_local_pose,
    get_grip_goal_dist,
    get_grip_goal_rot_diff,
)
from ..prtpr_model import PrtprModel

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.controllers.differential_ik import (
        DifferentialIKControllerCfg,
        DifferentialIKController,
    )


class ArmAction(ActionTerm):
    r"""Pre-trained PRTPR policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: ArmActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ArmActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[self.cfg.asset_name]

        # IK controller init
        self.diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            self.diff_ik_cfg, num_envs=env.num_envs, device=env.device
        )
        self.robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1  # type: ignore

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        self.policy = PrtprModel(cfg.policy_path)
        self.policy.reset()

        # raw action存放buffer
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        # low_level observations
        @configclass
        class LowLevelObsCfg:
            """Observation specifications for the low level MDP."""

            @configclass
            class LlPolicy(ObsGroup):
                """Observations for ll_policy group."""

                grip_point_pose_l = ObsTerm(func=get_grip_point_local_pose)
                goal_pose_l = ObsTerm(func=get_subgoal_local_pose)
                dist = ObsTerm(func=get_grip_goal_dist)
                rot_diff = ObsTerm(func=get_grip_goal_rot_diff)

                def __post_init__(self):
                    self.enable_corruption = True
                    self.concatenate_terms = True

            ll_policy: LlPolicy = LlPolicy()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager(LowLevelObsCfg, env)

        # ll actions limits
        self.ll_action_limits = torch.tensor(
            [
                [-math.pi, math.pi],
                [-math.pi / 2, math.pi / 2],
                [-0.02, 0.02],
                [-math.pi / 90, math.pi / 90],
            ],
            dtype=torch.float32,
            device=env.device,
        )
        self.ll_action_lower_limits, self.ll_action_upper_limits = torch.t(
            self.ll_action_limits
        )

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """将原始动作与预训练动作进行掩码操作，输出操作动作"""
        self._raw_actions[:] = actions

        low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
        self.tar_ee_pos, self.tar_ee_rot = self._resolve_target_ee_pose(low_level_obs)  # type: ignore
        tar_hand_pos, tar_hand_rot = self._resolve_target_hand_pose(
            self.tar_ee_pos, self.tar_ee_rot
        )

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
        ]
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        curr_hand_pos = (
            self.robot.data.body_pos_w[:, self.robot_entity_cfg.body_ids[0]]  # type: ignore
            - self._env.scene.env_origins
        )
        curr_hand_rot = self.robot.data.body_quat_w[
            :, self.robot_entity_cfg.body_ids[0]  # type: ignore
        ]
        # compute the joint commands
        self.diff_ik_controller.set_command(torch.cat((tar_hand_pos, tar_hand_rot)))
        self.joint_pos_des = self.diff_ik_controller.compute(
            curr_hand_pos, curr_hand_rot, jacobian, joint_pos
        )

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            self.robot.set_joint_position_target(
                self.joint_pos_des, self.robot_entity_cfg.joint_ids
            )
            self._counter = 0
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "next_ee_goal_pose"):
                # next goal
                marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
                marker_cfg.prim_path = "/Visuals/Actions/next_ee_goal"
                marker_cfg.markers["frame"].scale = (0.5, 0.5, 0.5)
                self.next_ee_goal_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.next_ee_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "next_ee_goal_pose"):
                self.next_ee_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        pos = self.tar_ee_pos + self._env.scene.env_origins
        rot = self.tar_ee_rot
        # display markers
        self.next_ee_goal_visualizer.visualize(pos, rot)

    """
    Internal helpers.
    """

    def _resolve_target_ee_pose(
        self,
        low_level_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute next gripper frame pos and rot."""
        curr_ee_pos, curr_ee_rot = low_level_obs[:, :3], low_level_obs[:, 3:7]  # type: ignore
        subgoal_rot = low_level_obs[:, 10:14]  # type: ignore

        rot_diff = normalize(quat_mul(subgoal_rot, quat_conjugate(curr_ee_rot)))
        w = rot_diff[:, 0]
        a = rot_diff[:, 1]
        b = rot_diff[:, 2]
        c = rot_diff[:, 3]
        rot_axis = torch.cat(
            [
                torch.reshape(a / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
                torch.reshape(b / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
                torch.reshape(c / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
            ],
            dim=-1,
        )

        ll_policy_action = (
            self.policy.get_action(low_level_obs).clone().clamp(-1.0, 1.0)
        )
        ll_policy_action = unscale_transform(
            ll_policy_action,
            self.ll_action_lower_limits,
            self.ll_action_upper_limits,
        )

        # ll_policy_action mask
        self._processed_actions = ll_policy_action * self._raw_actions

        thetas1 = self._processed_actions[:, 0]
        thetas2 = self._processed_actions[:, 1]
        dist = torch.abs(self._processed_actions[:, 2])

        _x = dist * torch.cos(thetas2) * torch.cos(thetas1)
        _y = dist * torch.cos(thetas2) * torch.sin(thetas1)
        _z = dist * torch.sin(thetas2)

        tar_ee_pos = curr_ee_pos + torch.stack((_x, _y, _z), dim=-1)

        angle = self._processed_actions[:, 3]
        rot = quat_from_angle_axis(angle, rot_axis)
        tar_ee_rot = quat_mul(rot, curr_ee_rot)

        return tar_ee_pos, tar_ee_rot

    def _resolve_target_hand_pose(
        self,
        tar_ee_pos: torch.Tensor,
        tar_ee_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the next gripper frame pos and rot to hand frame pos and rot."""
        gripper_frame_in_hand_frame = (0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0)
        hand_pos_to_gripper = -torch.tensor(
            gripper_frame_in_hand_frame[0:3], device=self.device
        ).repeat(self._env.num_envs, 1)
        hand_rot_to_gripper = quat_conjugate(
            torch.tensor(gripper_frame_in_hand_frame[3:7], device=self.device).repeat(
                self._env.num_envs, 1
            )
        )

        tar_hand_pos, tar_hand_rot = combine_frame_transforms(
            t01=tar_ee_pos,
            q01=tar_ee_rot,
            t12=hand_pos_to_gripper,
            q12=hand_rot_to_gripper,
        )

        return tar_hand_pos, tar_hand_rot


@configclass
class ArmActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`ArmActionCfg` for more details.
    """

    class_type: type[ActionTerm] = ArmAction
    """ Class of the action term."""
    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING  # type: ignore
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""


class GripperAction(ActionTerm):
    r"""gripper action term.

    gripper action term only has two target: gripper open / close.

    """

    cfg: GripperActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: GripperActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[self.cfg.asset_name]

        self.robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_finger_joint.*"]
        )

        # raw action存放buffer
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        self.open_pos = torch.full_like(
            self._raw_actions, fill_value=0.05, device=self.device
        )

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """将原始动作与预训练动作进行掩码操作，输出操作动作"""
        self._raw_actions[:] = actions
        self.joint_target_pos = self.open_pos * self._raw_actions
        self.joint_target_pos.repeat(1, 2)

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            self.robot.set_joint_position_target(
                self.joint_target_pos, self.robot_entity_cfg.joint_ids
            )
            self._counter = 0
        self._counter += 1


@configclass
class GripperActionCfg(ActionTermCfg):
    """Configuration for gripper action term.

    See :class:`GripperAction` for more details.
    """

    class_type: type[ActionTerm] = GripperAction
    """ Class of the action term."""
    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
