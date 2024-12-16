from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence, Literal
from dataclasses import MISSING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs.mdp import SceneEntityCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.managers.observation_manager import (
    ObservationGroupCfg as ObsGroup,
    ObservationManager,
)
from omni.isaac.lab.managers.observation_manager import ObservationTermCfg as ObsTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path
from .observations import (
    get_ee_local_pose,
    get_ee_subgoal_dist,
    get_ee_subgoal_rot_dist,
    get_arm_dof_pos,
)
from omni.isaac.lab.envs.mdp.observations import generated_commands
from ...prtpr_model import PrtprModel
from omni.isaac.lab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from omni.isaac.lab.utils.math import combine_frame_transforms


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class PreTrainedArmAction(ActionTerm):
    r"""Pre-trained prtpr policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: PreTrainedArmActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedArmActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[self.cfg.asset_name]

        # Ik contorller
        self.diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            self.diff_ik_cfg, num_envs=env.num_envs, device=env.device
        )

        self.robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        self.robot_entity_cfg.resolve(env.scene)
        self.hand_link_idx = self.robot_entity_cfg.body_ids[0]  # type: ignore
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1  # type: ignore

        # joint limited
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(
            device=self.device
        )
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(
            device=self.device
        )

        # robot speed scale
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits[:7])

        # hand-ee relationship
        self.hand_pos_in_ee = torch.tensor([0.0, 0.0, -0.09], device=env.device).repeat(
            self.num_envs, 1
        )
        self.hand_rot_in_ee = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        self.policy = PrtprModel(cfg.policy_path)

        # reset policy
        self.policy.reset()
        if self.policy.is_rnn:
            self.policy.init_rnn()

        # raw action存放buffer
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        self.action_scale = torch.tensor(
            self.policy.env_cfg["action_scale"],  # type: ignore
            device=self.device,
        )

        self.control_mode = cfg.mode

        # remap some of the low level observations to internal observations
        @configclass
        class LowLevelObsCfg:
            """Observation specifications for the low level MDP."""

            @configclass
            class LlPolicy(ObsGroup):
                """Observations for ll_policy group."""

                ee_pose_l = ObsTerm(
                    func=get_ee_local_pose,
                    params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
                )
                goal_pose_l = ObsTerm(
                    func=generated_commands, params={"command_name": "subgoals"}
                )
                dist = ObsTerm(
                    func=get_ee_subgoal_dist,
                    params={
                        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                        "subgoal_cmd_name": "subgoals",
                    },
                )
                rot_dist = ObsTerm(
                    func=get_ee_subgoal_rot_dist,
                    params={
                        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                        "subgoal_cmd_name": "subgoals",
                    },
                )
                arm_dof_pos = ObsTerm(
                    func=get_arm_dof_pos,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=["panda_joint.*"]
                        )
                    },
                )

                def __post_init__(self):
                    self.enable_corruption = False
                    self.concatenate_terms = True

            ll_policy: LlPolicy = LlPolicy()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager(
            {"ll_policy": LowLevelObsCfg().ll_policy}, env
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

        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: move, false: stay
            binary_mask = actions > 0

        # ptp agent actions
        low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

        self.joint_pos_des = self._resolve_low_level_actions(low_level_obs)  # type: ignore
        self._processed_actions = torch.where(
            binary_mask, self.joint_pos_des, self.robot.data.joint_pos[:, :7]
        )

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            self.robot.set_joint_position_target(
                self.joint_pos_des, self.robot_entity_cfg.joint_ids
            )
            self._counter = 0
        self._counter += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Internal helpers.
    """

    def _resolve_low_level_actions(
        self,
        low_level_obs: torch.Tensor,
    ) -> torch.Tensor:
        """compute low level actions."""
        # obs metrics
        current_dist = low_level_obs[:, 15]
        current_arm_dof_pos = low_level_obs[:, -7:]

        # ptp agent action
        if self.policy.has_batch_dimension is False:
            self.policy.get_batch_size(low_level_obs)

        ll_policy_action = (
            self.policy.get_action(low_level_obs).clone().clamp(-1.0, 1.0)
        )

        arm_targets1 = (
            current_arm_dof_pos
            + self.robot_dof_speed_scales[:7]
            * self._env.step_dt
            * ll_policy_action
            * self.action_scale
        )

        arm_targets = torch.clamp(
            arm_targets1,
            self.robot_dof_lower_limits[:7],
            self.robot_dof_upper_limits[:7],
        )

        # ik action
        if self.control_mode == "precision":
            curr_hand_pos, curr_hand_rot = (
                (
                    self.robot.data.body_pos_w[:, self.hand_link_idx, :]
                    - self._env.scene.env_origins
                ),
                self.robot.data.body_quat_w[:, self.hand_link_idx, :],
            )

            target_ee_pos, target_ee_rot = (
                low_level_obs[:, 7:10],
                low_level_obs[:, 10:14],
            )
            target_hand_pos, target_hand_rot = combine_frame_transforms(
                target_ee_pos, target_ee_rot, self.hand_pos_in_ee, self.hand_rot_in_ee
            )

            jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
            ]

            self.diff_ik_controller.set_command(
                command=torch.cat((target_hand_pos, target_hand_rot), dim=-1)
            )
            ik_arm_targets = self.diff_ik_controller.compute(
                curr_hand_pos, curr_hand_rot, jacobian, current_arm_dof_pos
            )

            ik_arm_targets = torch.clamp(
                ik_arm_targets,
                self.robot_dof_lower_limits[:7],
                self.robot_dof_upper_limits[:7],
            )

        arm_targets = torch.where(
            current_dist.unsqueeze(0).view(-1, 1) >= 0.05,
            arm_targets,
            ik_arm_targets,
        )

        return arm_targets


@configclass
class PreTrainedArmActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`ArmActionCfg` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedArmAction
    """ Class of the action term."""
    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING  # type: ignore
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 1
    """Decimation factor for the low level action term."""
    mode: Literal["common", "precision"] = MISSING  # type: ignore
