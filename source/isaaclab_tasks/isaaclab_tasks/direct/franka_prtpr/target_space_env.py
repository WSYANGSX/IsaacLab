from __future__ import annotations

import torch
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKControllerCfg, DifferentialIKController
from isaaclab.utils.math import (
    sample_uniform,
    unscale_transform,
    quat_mul,
    normalize,
    quat_conjugate,
    combine_frame_transforms,
)
from local.utils.math import rotation_distance, calculate_angle_between_vectors
from isaacsim.core.utils.torch import torch_rand_float, quat_from_angle_axis


@configclass
class TargetSpaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12  # 720 timesteps
    decimation = 4
    actions_space = 4
    observations_space = 23
    states_space = 0
    asymmetric_obs = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # target
    target: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visual/marker1",
        markers={
            "target": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )
        },
    )

    # target
    ee_frame: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visual/marker2",
        markers={
            "ee_frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )
        },
    )

    # ik controller
    ik_controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )

    # reward scales
    dist_reward_scale = -5.0
    rot_reward_scale = 0.1
    direction_change_penalty_scale = -1
    rot_eps = 0.1
    action_penalty_scale = -0.0001
    reach_target_bonus = 500
    eposide_lengths_penalty_scale = -0.001
    dist_tolerance = 0.1
    rot_tolerance = 0.1


class TargetSpaceEnv(DirectRLEnv):
    # reset()
    #   |-- _reset_index()          # _compute_intermediate_values, reset all envs
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()            # _compute_intermediate_values
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)     # _compute_intermediate_values
    #   |-- _get_observations()

    cfg: TargetSpaceEnvCfg

    def __init__(self, cfg: TargetSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.action_limits = torch.tensor(
            [
                [-torch.pi, torch.pi],
                [-torch.pi / 2, torch.pi / 2],
                [-0.04, 0.04],
                [-torch.pi / 18, torch.pi / 18],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.action_lower_limits, self.action_upper_limits = torch.t(self.action_limits.to(self.device))

        # visualization markers
        self.target = VisualizationMarkers(self.cfg.target)
        self.ee_frame = VisualizationMarkers(self.cfg.ee_frame)

        # task space controller
        self.ik_controller = DifferentialIKController(
            self.cfg.ik_controller, num_envs=self.num_envs, device=self.device
        )

        # robot propertities
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)

        self.hand_link_idx = self.robot_entity_cfg.body_ids[0]  # type: ignore
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1  # type: ignore

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
        ]

        self.ee_pos_in_hand = torch.tensor([0.0, 0.0, 0.09], device=self.device).repeat(self.num_envs, 1)
        self.ee_rot_in_hand = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # buffers
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)

        # robot relative
        self.robot_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_hand_pos_target = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_hand_rot_target = torch.zeros((self.num_envs, 4), device=self.device)

        self.last_robot_hand_pos = torch.zeros_like(self.robot_hand_pos)

        self.robot_ee_pos = torch.zeros_like(self.robot_hand_pos)
        self.robot_ee_rot = torch.zeros_like(self.robot_hand_rot)

        # target relative
        self.target_pos = torch.zeros_like(self.robot_hand_pos)
        self.target_rot = torch.zeros_like(self.robot_hand_rot)

        # direction
        self.last_direction = torch.zeros((self.num_envs, 3), device=self.device)
        self.current_direction = torch.zeros_like(self.last_direction)

        self.rot_axis = torch.zeros((self.num_envs, 3), device=self.device)

        self.ik_command = torch.zeros((self.num_envs, 7), device=self.device)

        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        # successes tracker
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_three_successes_rate = deque(maxlen=3)

        print("*************** task initialed *****************")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

        actions = unscale_transform(self.actions, self.action_lower_limits, self.action_upper_limits)

        thetas1 = actions[:, 0]
        thetas2 = actions[:, 1]
        dist = torch.abs(actions[:, 2])

        _x = dist * torch.cos(thetas2) * torch.cos(thetas1)
        _y = dist * torch.cos(thetas2) * torch.sin(thetas1)
        _z = dist * torch.sin(thetas2)

        self.robot_hand_pos_target = self.robot_hand_pos + torch.stack((_x, _y, _z), dim=-1)

        self.current_direction = normalize(torch.stack((_x, _y, _z), dim=-1))

        angle = actions[:, 3]
        rot = quat_from_angle_axis(angle, self.rot_axis)
        self.robot_hand_rot_target = quat_mul(rot, self.robot_hand_rot)

        self.ik_command = torch.cat((self.robot_hand_pos_target, self.robot_hand_rot_target), dim=-1)
        self.ik_controller.set_command(self.ik_command)

        # compute desire joint pos
        robot_dof_targets = self.ik_controller.compute(
            self.robot_hand_pos,
            self.robot_hand_rot,
            self.jacobian,
            self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids],
        )
        self.robot_dof_targets[:, self.robot_entity_cfg.joint_ids] = torch.clamp(
            robot_dof_targets,
            self.robot_dof_lower_limits[self.robot_entity_cfg.joint_ids],
            self.robot_dof_upper_limits[self.robot_entity_cfg.joint_ids],
        )

        self.last_actions = self.actions

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # no terminated reset
        terminated = torch.zeros_like(truncated)

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        total_reward, self.successes, self.reset_goal_buf = self._compute_rewards(
            self.reset_goal_buf,
            self.episode_length_buf,
            self.successes,
            self.robot_ee_pos,
            self.robot_ee_rot,
            self.target_pos,
            self.target_rot,
            self.last_direction,
            self.current_direction,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.cfg.action_penalty_scale,
            self.cfg.eposide_lengths_penalty_scale,
            self.cfg.direction_change_penalty_scale,
            self.cfg.reach_target_bonus,
            self.cfg.dist_tolerance,
            self.cfg.rot_tolerance,
        )

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _curriculum(self):
        # 改变训练成功的难度
        succecc_num = torch.sum(self.successes)
        success_rate = succecc_num / self.num_envs
        self.last_three_successes_rate.append(success_rate)

        last_three_successes_rate = torch.tensor(list(self.last_three_successes_rate), device=self.device)
        print("last three successes rate: ", last_three_successes_rate)

        if all(last_three_successes_rate >= 2.0):
            print("******************** curriculum performed **************************")
            self.cfg.dist_tolerance *= 0.8

            if self.cfg.dist_tolerance < 0.005:
                self.cfg.dist_tolerance = 0.005

        print("current dist_tolerance: ", self.cfg.dist_tolerance)
        print("current rot_tolerance: ", self.cfg.rot_tolerance)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)  # type: ignore
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),  # type: ignore
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)  # type: ignore
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)  # type: ignore

        # target state
        self._reset_target_pose(env_ids)

        # curriculum
        self._curriculum()
        self.successes[env_ids] = 0

        # compute physx data
        self.sim.step(render=False)
        self.scene.update(self.cfg.sim.dt)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        # ik controller
        self.ik_controller.reset(env_ids)  # type:ignore
        self.ik_command[env_ids] = torch.cat((self.robot_hand_pos[env_ids], self.robot_hand_rot[env_ids]), dim=-1)
        self.ik_controller.set_command(self.ik_command)

    def _reset_target_pose(self, env_ids):
        # reset target position
        new_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_pos[:, 0] = torch.abs(new_pos[:, 0]) * 0.3 + 0.35
        new_pos[:, 1] = new_pos[:, 1] * 0.2
        new_pos[:, 2] = torch.abs(new_pos[:, 2]) * 0.35 + 0.15

        # reset target rotation
        new_rot = generate_random_quat_with_z_in_hyposphere(len(env_ids), device=self.device)

        # update target pose
        self.target_pos[env_ids] = new_pos
        self.target_rot[env_ids] = new_rot

        self.target.visualize(self.target_pos + self.scene.env_origins, self.target_rot)

        self.reset_goal_buf[env_ids] = 0

    def _get_observations(self) -> dict:
        arm_dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )[:, :7]

        dist = torch.norm(self.target_pos - self.robot_hand_pos, p=2, dim=-1, keepdim=True)

        rot_dist = rotation_distance(self.target_rot, self.robot_hand_rot)
        rot_dist = torch.unsqueeze(rot_dist, dim=0).view(-1, 1)

        obs = torch.cat(
            (
                self.robot_hand_pos,
                self.robot_hand_rot,
                self.target_pos,
                self.target_rot,
                dist,
                rot_dist,
                arm_dof_pos_scaled,
            ),
            dim=-1,
        )

        observations = {"policy": obs}

        if self.cfg.asymmetric_obs:
            states = self._get_states()
            observations = {"policy": obs, "critic": states}

        return observations

    # auxiliary methods
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.robot_hand_pos[env_ids] = (
            self._robot.data.body_pos_w[env_ids, self.hand_link_idx] - self.scene.env_origins[env_ids]
        )
        self.robot_hand_rot[env_ids] = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        self.last_direction = normalize(self.robot_hand_pos - self.last_robot_hand_pos)
        self.last_robot_hand_pos[env_ids] = self.robot_hand_pos[env_ids]

        self.robot_ee_pos, self.robot_ee_rot = combine_frame_transforms(
            self.robot_hand_pos,
            self.robot_hand_rot,
            self.ee_pos_in_hand,
            self.ee_rot_in_hand,
        )

        self.ee_frame.visualize(
            self.robot_ee_pos + self.scene.env_origins,
            self.robot_ee_rot,
        )

        rot_diff = normalize(quat_mul(self.target_rot[env_ids], quat_conjugate(self.robot_ee_rot[env_ids])))
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
        self.rot_axis[env_ids] = rot_axis

        self.jacobian[env_ids] = self._robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
        ][env_ids]

    def _compute_rewards(
        self,
        reset_goal_buf: torch.Tensor,
        eposide_length_buf: torch.Tensor,
        successes: torch.Tensor,
        franka_ee_pos: torch.Tensor,
        franka_ee_rot: torch.Tensor,
        target_pos: torch.Tensor,
        target_rot: torch.Tensor,
        last_direction: torch.Tensor,
        curr_direction: torch.Tensor,
        dist_reward_scale: float,
        rot_reward_scale: float,
        rot_eps: float,
        action_penalty_scale: float,
        eposide_length_penalty_scale: float,
        direction_change_penalty_scale: float,
        reach_target_bonus: float,
        dist_tolerance: float,
        rot_tolerance: float,
    ):
        # distance from ee to the target
        target_dist = torch.norm(franka_ee_pos - target_pos, p=2, dim=-1)
        rot_dist = rotation_distance(franka_ee_rot, target_rot)

        dist_rew = target_dist * dist_reward_scale
        rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        # regularization on the actions (summed for each environment)
        action_rew = torch.sum(self._robot.data.joint_acc[:, :7] ** 2, dim=-1) * action_penalty_scale

        # direction change angle in rad
        direction_change_angle = calculate_angle_between_vectors(last_direction, curr_direction).view(
            -1,
        )
        direction_change_rew = direction_change_angle * direction_change_penalty_scale

        # eposide_length 相关惩罚
        eposide_length_penalty = eposide_length_buf * eposide_length_penalty_scale

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        reward = (
            dist_rew + rot_rew + action_rew + eposide_length_penalty + direction_change_rew
        )  # reward的shape为(num_envs, )

        # Find out which envs hit the target and update successes count
        dist_successes = torch.where(
            target_dist <= dist_tolerance,
            torch.ones_like(reset_goal_buf),
            torch.zeros_like(reset_goal_buf),
        )
        rot_successes = torch.where(
            torch.abs(rot_dist) <= rot_tolerance,
            torch.ones_like(reset_goal_buf),
            torch.zeros_like(reset_goal_buf),
        )
        target_resets = torch.where(
            dist_successes + rot_successes == 2,
            torch.ones_like(reset_goal_buf),
            torch.zeros_like(reset_goal_buf),
        )

        successes = successes + target_resets

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward = torch.where(target_resets == 1, reward + reach_target_bonus, reward)

        return reward, successes, target_resets


class TargetSpacePlayEnv(TargetSpaceEnv):
    def __init__(
        self,
        cfg: TargetSpaceEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.dist_tolerance = 0.01
        self.cfg.episode_length_s = 12

    def _curriculum(self):
        pass

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

        self.actions[:, 3] = torch.where(
            torch.abs(rotation_distance(self.robot_hand_rot, self.target_rot)) <= self.cfg.rot_tolerance,
            torch.zeros_like(self.actions[:, 3]),
            self.actions[:, 3].clone(),
        )

        actions = unscale_transform(self.actions, self.action_lower_limits, self.action_upper_limits)

        thetas1 = actions[:, 0]
        thetas2 = actions[:, 1]
        dist = torch.abs(actions[:, 2])

        _x = dist * torch.cos(thetas2) * torch.cos(thetas1)
        _y = dist * torch.cos(thetas2) * torch.sin(thetas1)
        _z = dist * torch.sin(thetas2)

        self.robot_hand_pos_target = self.robot_hand_pos + torch.stack((_x, _y, _z), dim=-1)

        self.curr_direction = torch.stack((_x, _y, _z), dim=-1)

        angle = actions[:, 3]
        rot = quat_from_angle_axis(angle, self.rot_axis)
        self.robot_hand_rot_target = quat_mul(rot, self.robot_hand_rot)

        self.ik_command = torch.cat((self.robot_hand_pos_target, self.robot_hand_rot_target), dim=-1)
        self.ik_controller.set_command(self.ik_command)

        # compute desire joint pos
        robot_dof_targets = self.ik_controller.compute(
            self.robot_hand_pos,
            self.robot_hand_rot,
            self.jacobian,
            self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids],
        )
        self.robot_dof_targets[:, self.robot_entity_cfg.joint_ids] = torch.clamp(
            robot_dof_targets,
            self.robot_dof_lower_limits[self.robot_entity_cfg.joint_ids],
            self.robot_dof_upper_limits[self.robot_entity_cfg.joint_ids],
        )


"""
Helper function
"""


def generate_random_quat_with_z_in_hyposphere(num_samples: int, device: str):
    rot_x = quat_from_angle_axis(
        torch.rand(num_samples, device=device) * torch.pi + torch.pi / 2,
        torch.tensor((1.0, 0.0, 0.0), device=device).repeat(num_samples, 1),
    )
    rot_y = quat_from_angle_axis(
        (torch.rand(num_samples, device=device) * 2 - 1) * torch.pi / 2,
        torch.tensor((0.0, 1.0, 0.0), device=device).repeat(num_samples, 1),
    )
    rot_z = quat_from_angle_axis(
        torch.rand(num_samples, device=device),
        torch.tensor((0.0, 0.0, 1.0), device=device).repeat(num_samples, 1),
    )

    return quat_mul(rot_z, quat_mul(rot_y, rot_x))
