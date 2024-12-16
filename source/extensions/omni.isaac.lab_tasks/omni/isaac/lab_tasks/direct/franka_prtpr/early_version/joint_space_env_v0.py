from __future__ import annotations

import torch
from collections import deque

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import (
    sample_uniform,
    combine_frame_transforms,
    quat_from_angle_axis,
    quat_mul,
)
from my_projects.utils.math import rotation_distance
from omni.isaac.core.utils.torch import torch_rand_float


@configclass
class JointSpaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12  # 720 timesteps
    decimation = 4
    num_actions = 7
    num_observations = 23
    num_states = 0
    asymmetric_obs = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

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

    action_scale = [5.0, 5.0, 5.0, 2.5, 2.5, 2.5, 2.5]

    # reward scales
    dist_reward_scale = -5.0
    rot_reward_scale = 0.1
    direction_change_penalty_scale = -1
    rot_eps = 0.1
    action_penalty_scale = -0.1
    reach_target_bonus = 500
    eposide_lengths_penalty_scale = -0.001
    dist_tolerance = 0.1
    rot_tolerance = 0.1


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
        self.ee_frame = VisualizationMarkers(self.cfg.ee_frame)

        # robot propertities
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)
        self.hand_link_idx = self.robot_entity_cfg.body_ids[0]  # type: ignore

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.ee_pos_in_hand = torch.tensor([0.0, 0.0, 0.09], device=self.device).repeat(self.num_envs, 1)
        self.ee_rot_in_hand = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.0
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.0

        # buffers
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.robot_dof_pos = torch.zeros_like(self.robot_dof_targets)

        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        # robot relative
        self.robot_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_hand_pos_target = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_hand_rot_target = torch.zeros((self.num_envs, 4), device=self.device)

        self.robot_ee_pos = torch.zeros_like(self.robot_hand_pos)
        self.robot_ee_rot = torch.zeros_like(self.robot_hand_rot)

        # target relative
        self.target_pos = torch.zeros_like(self.robot_hand_pos)
        self.target_rot = torch.zeros_like(self.robot_hand_rot)

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
        self.last_actions[:] = self.actions[:]
        self.actions = actions.clone().clamp(-1.0, 1.0)

        arm_targets = self.robot_dof_pos[:, :7] + self.robot_dof_speed_scales[
            :7
        ] * self.dt * self.actions * torch.tensor(self.cfg.action_scale, device=self.device)
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

        total_reward, self.successes, self.reset_goal_buf = self._compute_rewards(
            self.reset_goal_buf,
            self.episode_length_buf,
            self.successes,
            self.actions,
            self.last_actions,
            self.robot_ee_pos,
            self.robot_ee_rot,
            self.target_pos,
            self.target_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.cfg.action_penalty_scale,
            self.cfg.eposide_lengths_penalty_scale,
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
            self.cfg.dist_tolerance *= 0.6

            if self.cfg.dist_tolerance < 0.01:
                self.cfg.dist_tolerance = 0.01

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

        self.last_actions[:] = torch.zeros_like(self.last_actions)

    def _reset_target_pose(self, env_ids):
        # reset target position
        new_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_pos[:, 0] = torch.abs(new_pos[:, 0]) * 0.3 + 0.35
        new_pos[:, 1] = new_pos[:, 1] * 0.3
        new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.35) + 0.15

        # reset target rotation
        new_rot = generate_random_quat_with_z_in_hyposphere(len(env_ids), device=self.device)

        # update target pose
        self.target_pos[env_ids] = new_pos
        self.target_rot[env_ids] = new_rot

        self.target.visualize(self.target_pos + self.scene.env_origins, self.target_rot)

        self.reset_goal_buf[env_ids] = 0

    def _get_observations(self) -> dict:
        dist = torch.norm(self.target_pos - self.robot_ee_pos, p=2, dim=-1, keepdim=True)

        rot_dist = rotation_distance(self.target_rot, self.robot_ee_rot)
        rot_dist = torch.unsqueeze(rot_dist, dim=0).view(-1, 1)

        obs = torch.cat(
            (
                self.robot_ee_pos,
                self.robot_ee_rot,
                self.target_pos,
                self.target_rot,
                dist,
                rot_dist,
                self.robot_dof_pos,
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

        self.robot_dof_pos = self._robot.data.joint_pos[:, :7]

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

    def _compute_rewards(
        self,
        reset_goal_buf: torch.Tensor,
        eposide_length_buf: torch.Tensor,
        successes: torch.Tensor,
        actions: torch.Tensor,
        last_actions: torch.Tensor,
        franka_ee_pos: torch.Tensor,
        franka_ee_rot: torch.Tensor,
        target_pos: torch.Tensor,
        target_rot: torch.Tensor,
        dist_reward_scale: float,
        rot_reward_scale: float,
        rot_eps: float,
        action_penalty_scale: float,
        eposide_length_penalty_scale: float,
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
        action_rew = torch.norm(actions - last_actions, p=2, dim=-1) * action_penalty_scale

        # eposide_length 相关惩罚
        eposide_length_penalty = eposide_length_buf * eposide_length_penalty_scale

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        reward = dist_rew + rot_rew + action_rew + eposide_length_penalty  # reward的shape为(num_envs, )

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


class JointSpacePlayEnv(JointSpaceEnv):
    def __init__(
        self,
        cfg: JointSpaceEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg.dist_tolerance = 0.02
        self.cfg.episode_length_s = 12

    def _curriculum(self):
        pass