from __future__ import annotations

import torch
from collections import deque

from gymnasium import spaces
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.sensors import FrameTransformerCfg, FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    sample_uniform,
    quat_from_angle_axis,
    quat_mul,
)
from isaacsim.core.utils.torch import torch_rand_float
from example.utils.math import rotation_distance, calculate_angle_between_vectors

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class JointSpaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 4
    action_space: spaces.Box = spaces.Box(low=-1, high=1, shape=(7,))
    observation_space: spaces.Box = spaces.Box(low=-torch.inf, high=torch.inf, shape=(33,))
    state_space = 0
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
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
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
        prim_path="/Visual/Target",
        markers={
            "target": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )
        },
    )

    # ee_frame
    marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg.replace(prim_path="/Visuals/EEFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
        ],
    )

    action_scale = [5.0, 5.0, 5.0, 2.5, 2.5, 2.5, 2.5]

    # reward weights
    dist_reward_weight = 5
    quat_reward_weight = 5
    dof_vel_penalty_weight = -1e-2
    dof_acc_penalty_weight = -1e-4
    ee_vel_direction_penalty_weight = -0.1
    reach_target_bonus = 500
    task_fail_penalty = -500
    episode_lengths_penalty_weight = -1e-3

    # reward threholds
    dist_tolerance = 0.08
    quat_tolerance = 0.1
    dist_std = 1.5
    quat_std = 1.5


class JointSpaceEnv(DirectRLEnv):
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

    cfg: JointSpaceEnvCfg

    def __init__(self, cfg: JointSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # visualization markers
        self.target = VisualizationMarkers(self.cfg.target)

        # robot propertities
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)
        self.hand_link_idx = self.robot_entity_cfg.body_ids[0]  # type: ignore

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits[:7])
        self.action_scale = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float32)

        # buffers
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device, dtype=torch.float32
        )

        self.actions = torch.zeros(
            (self.num_envs, *self.cfg.action_space.shape), device=self.device, dtype=torch.float32
        )  # type: ignore
        self.prev_actions = torch.zeros_like(self.actions)

        # target relative
        self.target_pos_l = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_quat_l = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.ee_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.prev_ee_lin_vel = torch.zeros_like(self.ee_lin_vel)

        # successes tracker
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_three_successes_rate = deque(maxlen=3)

        print("[INFO] *************** task initialized *****************")

    def _setup_scene(self):
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # ee_frame
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.scene.sensors["ee_frame"] = self._ee_frame

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
        self.prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(self.cfg.action_space.low[-1], self.cfg.action_space.high[-1])

        arm_targets = (
            self._robot.data.joint_pos[:, :7] + self.actions * self.dt * self.robot_dof_speed_scales * self.action_scale
        )

        self.robot_dof_targets[:, :7] = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits[:7],
            self.robot_dof_upper_limits[:7],
        )

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

        total_reward, self.successes, self.reset_goal_buf = compute_rewards(
            self.ee_pos_l,
            self.ee_quat_l,
            self.target_pos_l,
            self.target_quat_l,
            self.robot_dof_vel,
            self.robot_dof_acc,
            self.ee_lin_vel,
            self.prev_ee_lin_vel,
            self.cfg.dist_reward_weight,
            self.cfg.quat_reward_weight,
            self.cfg.dof_vel_penalty_weight,
            self.cfg.dof_acc_penalty_weight,
            self.cfg.ee_vel_direction_penalty_weight,
            self.cfg.episode_lengths_penalty_weight,
            self.cfg.reach_target_bonus,
            self.cfg.task_fail_penalty,
            self.cfg.dist_std,
            self.cfg.quat_std,
            self.cfg.dist_tolerance,
            self.cfg.quat_tolerance,
            self.successes,
            self.reset_goal_buf,
            self.episode_length_buf,
            self.max_episode_length,
        )

        # reset goals if the goal has been reached
        reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_goal_env_ids) > 0:
            self._reset_target_pose(reset_goal_env_ids)

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
            self.cfg.dist_tolerance *= 0.9
            self.cfg.quat_tolerance *= 0.8

            if self.cfg.dist_tolerance < 0.005:
                self.cfg.dist_tolerance = 0.005
            if self.cfg.quat_tolerance < 0.08:
                self.cfg.quat_tolerance = 0.08

        print("[INFO] Current dist_tolerance: ", self.cfg.dist_tolerance)
        print("[INFO] Current rot_tolerance: ", self.cfg.quat_tolerance)

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

        # # compute physx data
        # self.sim.step(render=False)
        # self.scene.update(self.cfg.sim.dt)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()

        self.prev_actions[env_ids] = 0
        self.actions[env_ids] = 0

        self.ee_lin_vel[env_ids] = 0
        self.prev_ee_lin_vel[env_ids] = 0

    def _reset_target_pose(self, env_ids):
        # reset target position
        new_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_pos[:, 0] = torch.abs(new_pos[:, 0]) * 0.3 + 0.35
        new_pos[:, 1] = new_pos[:, 1] * 0.3
        new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.35) + 0.15

        # reset target rotation
        new_quat = generate_random_quat_with_z_in_hyposphere(len(env_ids), device=self.device)

        # update target pose
        self.target_pos_l[env_ids] = new_pos
        self.target_quat_l[env_ids] = new_quat

        self.target.visualize(self.target_pos_l + self.scene.env_origins, self.target_quat_l)

        self.reset_goal_buf[env_ids] = 0

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.ee_pos_l,
                self.ee_quat_l,
                self.target_pos_l,
                self.target_quat_l,
                self.ee_target_dist,
                self.ee_target_quat_dist,
                self.robot_dof_pos,
                self.robot_dof_vel,
                self.ee_lin_vel,
            ),
            dim=-1,
        )

        return {"policy": obs}

    # auxiliary methods
    def _compute_intermediate_values(self):
        self.robot_dof_pos = self._robot.data.joint_pos[:, :7]
        self.robot_dof_vel = self._robot.data.joint_vel[:, :7]
        self.robot_dof_acc = self._robot.data.joint_acc[:, :7]

        self.ee_pos_l, self.ee_quat_l = (
            self._ee_frame.data.target_pos_source[:, 0, :],
            self._ee_frame.data.target_quat_source[:, 0, :],
        )

        self.prev_ee_lin_vel[:] = self.ee_lin_vel
        self.ee_lin_vel[:] = self._robot.data.body_lin_vel_w[:, self.hand_link_idx, :]

        self.ee_target_dist = torch.norm(self.target_pos_l - self.ee_pos_l, p=2, dim=-1, keepdim=True)
        self.ee_target_quat_dist = rotation_distance(self.target_quat_l, self.ee_quat_l).view(-1, 1)


"""
Helper function
"""


@torch.jit.script
def compute_rewards(
    ee_pos_l: torch.Tensor,
    ee_quat_l: torch.Tensor,
    target_pos_l: torch.Tensor,
    target_quat_l: torch.Tensor,
    robot_dof_vel: torch.Tensor,
    robot_dof_acc: torch.Tensor,
    ee_lin_vel: torch.Tensor,
    prev_ee_lin_vel: torch.Tensor,
    dist_reward_weight: float,
    quat_reward_weight: float,
    dof_vel_penalty_weight: float,
    dof_acc_penalty_weight: float,
    ee_vel_direction_penalty_weight: float,
    eposide_lengths_penalty_weight: float,
    reach_target_bonus: float,
    task_fail_penalty: float,
    dist_std: float,
    quat_std: float,
    dist_tolerance: float,
    quat_tolerance: float,
    successes: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    eposide_length_buf: torch.Tensor,
    max_eposide_length: int,
):
    # distance reward
    ee_target_dist = torch.norm(ee_pos_l - target_pos_l, p=2, dim=-1)
    ee_target_dist_reward = 1 - torch.tanh(ee_target_dist / dist_std)

    # quat reward
    ee_target_quat_dist = rotation_distance(ee_quat_l, target_quat_l)
    ee_target_quat_dist_reward = 1 - torch.tanh(ee_target_quat_dist / quat_std)

    # joint vel penalty
    dof_vel_penalty = torch.sum(torch.square(robot_dof_vel), dim=-1)

    # joint acc penalty
    dof_acc_penalty = torch.sum(torch.square(robot_dof_acc), dim=-1)

    # ee lin vel direction change penalty
    direction_angle = calculate_angle_between_vectors(ee_lin_vel, prev_ee_lin_vel).squeeze()
    direction_angle_penalty = torch.tanh(direction_angle)

    # eposide_length penalty
    eposide_length_penalty = eposide_length_buf

    # print("ee_target_dist_reward", ee_target_dist_reward * dist_reward_weight)
    # print("ee_target_quat_dist_reward", ee_target_quat_dist_reward * quat_reward_weight)
    # print("dof_vel_penalty", dof_vel_penalty * dof_vel_penalty_weight)
    # print("dof_acc_penalty", dof_acc_penalty * dof_acc_penalty_weight)
    # print("direction_angle_penalty", direction_angle_penalty * ee_vel_direction_penalty_weight)
    # print("eposide_length_penalty", eposide_length_penalty * eposide_lengths_penalty_weight)

    reward = (
        ee_target_dist_reward * dist_reward_weight
        + ee_target_quat_dist_reward * quat_reward_weight
        + dof_vel_penalty * dof_vel_penalty_weight
        + dof_acc_penalty * dof_acc_penalty_weight
        + direction_angle_penalty * ee_vel_direction_penalty_weight
        + eposide_length_penalty * eposide_lengths_penalty_weight
    )

    # Find out which envs hit the target and update successes count
    task_complete = (ee_target_dist <= dist_tolerance) & (ee_target_quat_dist <= quat_tolerance)

    target_resets = torch.where(
        task_complete,
        torch.ones_like(reset_goal_buf),
        torch.zeros_like(reset_goal_buf),
    )

    successes = successes + target_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(task_complete, reward + reach_target_bonus, reward)

    # fail penalty
    task_fail = ((ee_target_dist > dist_tolerance) | (ee_target_quat_dist > quat_tolerance)) & (
        eposide_length_buf == max_eposide_length - 1
    )
    reward = torch.where(task_fail, reward + task_fail_penalty, reward)

    return reward, successes, target_resets


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


class JointSpacePlayEnv(JointSpaceEnv):
    def __init__(
        self,
        cfg: JointSpaceEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.dist_tolerance = 0.005
        self.cfg.quat_tolerance = 0.08
        self.cfg.episode_length_s = 12

    def _curriculum(self):
        pass
