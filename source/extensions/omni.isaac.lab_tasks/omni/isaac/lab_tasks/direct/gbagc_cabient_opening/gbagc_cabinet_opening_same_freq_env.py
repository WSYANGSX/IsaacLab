from __future__ import annotations

import torch

from typing import Literal

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sensors.frame_transformer import FrameTransformer, FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from my_projects.utils.math import rotation_distance

from .subogal_planner import SubgoalPlanner
from .prtpr_agent import PrtprAgent

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

# set print style
torch.set_printoptions(threshold=torch.inf)


@configclass
class GbagcCabinetOpeningEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 4
    action_space = 2
    observation_space = 38
    state_space = 0

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
                fix_root_link=True,
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
                velocity_limit=0.3,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"omniverse://localhost/Library/my_usd/cabinet/cabinet_instanceable.usd",
            activate_contact_sensors=False,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.42, 0.17)),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
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

    # lid_frame
    handle_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_handle_top",
                name="drawer_handle",
                offset=OffsetCfg(
                    pos=(0.29, 0.0, 0.01),
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                ),
            ),
        ],
    )

    # subgoal Visualization
    subgoal_visualization: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Subgoals",
        markers={
            "subgoals": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )
        },
    )

    # subgoals relative
    subgoals = {
        "handle": [
            [0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.36, 1.0, 0.0, 0.0, 0.0],
        ]
    }
    thresholds = {"handle": [[0.01, 0.0], [0.01, 0.0], [0.02, 0.0]]}

    # reward scales
    subgoal_bonus = 10
    task_complete_bonus = 2
    action_penalty_weight = -1e-4

    # subgoal control mode
    subgoal_control_mode: Literal["position", "pose"] = "position"


class GbagcCabinetOpeningEnv(DirectRLEnv):
    # reset()
    #   |-- _reset_index()                 reset all envs, _compute_intermediate_values
    #   |-- _get_observations()
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()                   _compute_intermediate_values
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)            _compute_intermediate_values
    #   |-- _get_observations()

    cfg: GbagcCabinetOpeningEnvCfg

    def __init__(self, cfg: GbagcCabinetOpeningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.subgoal_control_mode = self.cfg.subgoal_control_mode

        # robot relative propertities
        self.arm_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
        self.arm_entity_cfg.resolve(self.scene)
        self.gripper_entity_cfg.resolve(self.scene)

        self.arm_joint_idx = self.arm_entity_cfg.joint_ids  # type: ignore
        self.gripper_joint_idx = self.gripper_entity_cfg.joint_ids  # type: ignore
        self.hand_link_idx = self.arm_entity_cfg.body_ids[0]  # type: ignore

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits[: len(self.arm_joint_idx)])  # type: ignore

        self.gripper_open_command = self.robot_dof_upper_limits[-1]
        self.gripper_close_command = self.robot_dof_lower_limits[-1]

        # cabinet
        self.drawer_joint_idx = self._cabinet.find_joints("drawer_top_joint")[0][0]

        # buffers
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)  # type: ignore
        self.prev_actions = torch.zeros_like(self.actions)

        self.arm_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self.prev_arm_actions = torch.zeros_like(self.arm_actions)
        self.gripper_actions = torch.zeros_like(self.arm_actions)
        self.prev_gripper_actions = torch.zeros_like(self.arm_actions)

        # subgoals relative
        self.subgoal_planner = SubgoalPlanner(
            subgoals=self.cfg.subgoals, thresholds=self.cfg.thresholds, num_envs=self.num_envs, device=self.device
        )

        # low-level model
        checkpoint_path = (
            "/home/yangxf/my_projects/IsaacLab/logs/rl_games/franka_prtpr_jointspace_direct/v2/nn/best_model.pth"
        )
        self.prtpr_agent = PrtprAgent("Isaac-Franka_Prtpr-Direct-JointSpace-v0", self.device, checkpoint_path)

        self.action_scale = torch.tensor(self.prtpr_agent.env_cfg.action_scale, device=self.device)

        # subgoal visualization
        self.subgoal_visualization = VisualizationMarkers(self.cfg.subgoal_visualization)

        # success rate
        self.successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _setup_scene(self):
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # cabient
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["cabinet"] = self._cabinet

        # ground
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # ee_frame & lid_frame
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self._handle_frame = FrameTransformer(self.cfg.handle_frame)
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.sensors["handle_frame"] = self._handle_frame

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions[:] = torch.clamp(actions, min=-1, max=1)

        self.prev_arm_actions[:] = self.arm_actions
        self.prev_gripper_actions[:] = self.gripper_actions

        self.arm_actions[:] = torch.where(self.actions[:, 0] >= 0, 1, 0).view(-1, 1)
        self.gripper_actions[:] = torch.where(self.actions[:, 1] >= 0, 1, 0).view(-1, 1).view(-1, 1)

        ll_obs = self._get_low_level_observations()
        if self.prtpr_agent.has_batch_dimension is False:
            self.prtpr_agent.get_batch_size(ll_obs["ll_obs"])
        low_level_actions = self.prtpr_agent.get_action(ll_obs["ll_obs"])

        arm_targets = (
            self._robot.data.joint_pos[:, self.arm_joint_idx]
            + (self.robot_dof_speed_scales * self.dt * low_level_actions * self.action_scale) * self.arm_actions
        )

        gripper_targets = (
            torch.where(self.gripper_actions == 1, self.gripper_open_command, self.gripper_close_command)
            .view(-1, 1)
            .repeat(1, 2)
        )

        self.robot_dof_targets = torch.clamp(
            torch.cat((arm_targets, gripper_targets), dim=-1), self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        reward, self.successes = _compute_rewards(
            self.arm_actions,
            self.prev_arm_actions,
            self.gripper_actions,
            self.prev_gripper_actions,
            self.ee_subgoals_dist,
            self.ee_subgoals_quat_dist,
            self.threshold,
            self.drawer_dof_pos,
            self.cfg.subgoal_bonus,
            self.cfg.task_complete_bonus,
            self.cfg.action_penalty_weight,
            self.subgoal_control_mode,
            self.subgoal_planner.dones,
            self.successes,
        )

        return reward

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

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)  # type: ignore
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)  # type: ignore

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()

        # compute physx data
        self.sim.step(render=True)
        self.scene.update(self.cfg.sim.dt)

        # Reset subgoal planner
        self.subgoal_planner.reset(
            objects_pose={"handle": torch.cat((self.handle_pos_l[env_ids], self.handle_quat_l[env_ids]), dim=-1)},
            env_ids=env_ids,
        )

        self.prev_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0
        self.prev_gripper_actions[env_ids] = 0

        self.successes[env_ids] = 0

    def _get_low_level_observations(self) -> dict:
        ll_obs = torch.cat(
            (
                self.ee_pos_l,
                self.ee_quat_l,
                self.subgoal_pos_l,
                self.subgoal_quat_l,
                self.ee_subgoals_dist.unsqueeze(-1),
                self.ee_subgoals_quat_dist.unsqueeze(-1),
                self.robot_dof_pos[:, self.arm_joint_idx],
                self.robot_dof_vel[:, self.arm_joint_idx],
                self.ee_lin_vel,
            ),
            dim=-1,
        )
        return {"ll_obs": torch.clamp(ll_obs, -5, 5)}

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot_dof_pos,
                self.robot_dof_vel,
                self.ee_pos_l,
                self.ee_quat_l,
                self.handle_pos_l,
                self.handle_quat_l,
                self.drawer_dof_pos.unsqueeze(-1),
                self.drawer_dof_vel.unsqueeze(-1),
                self.ee_subgoals_dist.unsqueeze(-1),
                self.ee_subgoals_quat_dist.unsqueeze(-1),
                self.actions,
            ),
            dim=-1,
        )

        return {"policy": torch.clamp(obs, -5, 5)}

    # auxiliary methods

    def _compute_intermediate_values(self):
        self.robot_dof_pos = self._robot.data.joint_pos
        self.robot_dof_vel = self._robot.data.joint_vel

        self.drawer_dof_pos = self._cabinet.data.joint_pos[:, self.drawer_joint_idx]
        self.drawer_dof_vel = self._cabinet.data.joint_vel[:, self.drawer_joint_idx]

        self.ee_pos_l, self.ee_quat_l = (
            self._ee_frame.data.target_pos_source[:, 0, :],
            self._ee_frame.data.target_quat_source[:, 0, :],
        )

        self.handle_pos_l, self.handle_quat_l = (
            self._handle_frame.data.target_pos_source[:, 0, :],
            self._handle_frame.data.target_quat_source[:, 0, :],
        )

        self.ee_lin_vel = self._robot.data.body_lin_vel_w[:, self.hand_link_idx, :]

        self.subgoal_pos_l = self.subgoal_planner.current_subgoals[:, :3]
        self.subgoal_quat_l = self.subgoal_planner.current_subgoals[:, 3:7]
        self.threshold = self.subgoal_planner.current_thresholds

        self.ee_subgoals_dist = torch.norm(self.subgoal_pos_l - self.ee_pos_l, p=2, dim=-1)
        self.ee_subgoals_quat_dist = rotation_distance(self.ee_quat_l, self.subgoal_quat_l)

        self.subgoal_visualization.visualize(self.subgoal_pos_l + self.scene.env_origins, self.subgoal_quat_l)

        include_quat = True if self.subgoal_control_mode == "pose" else False
        self.subgoal_planner.step(
            curr_ee_pose=torch.cat((self.ee_pos_l, self.ee_quat_l), dim=-1), include_quat=include_quat
        )


@torch.jit.script
def _compute_rewards(
    arm_actions: torch.Tensor,
    prev_arm_actions: torch.Tensor,
    gripper_actions: torch.Tensor,
    prev_gripper_actions: torch.Tensor,
    ee_subgoal_dist: torch.Tensor,
    ee_subgoal_quat_dist: torch.Tensor,
    threshold: torch.Tensor,
    drawer_dof_pos: torch.Tensor,
    subgoal_bonus: float,
    task_complete_bonus: float,
    action_penalty_weight: float,
    subgoal_control_mode: str,
    subgoal_finished: torch.Tensor,
    successes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # action change penalty
    action_rate_penalty = (
        torch.sum(
            torch.square(
                torch.cat((arm_actions, gripper_actions), dim=-1)
                - torch.cat((prev_arm_actions, prev_gripper_actions), dim=-1)
            ),
            dim=-1,
        )
        * action_penalty_weight
    )

    # bonus for reaching subgoal
    if subgoal_control_mode == "pose":
        subgoal_rewards = torch.where(
            (ee_subgoal_dist <= threshold[:, 0]) & (ee_subgoal_quat_dist <= threshold[:, 1]) & (subgoal_finished == 0),
            subgoal_bonus,
            0,
        )
    else:
        subgoal_rewards = torch.where((ee_subgoal_dist <= threshold[:, 0]) & (subgoal_finished == 0), subgoal_bonus, 0)

    rewards = action_rate_penalty + subgoal_rewards

    # bonus for opening drawer
    rewards = torch.where(drawer_dof_pos > 0.01, rewards + 5, rewards)
    rewards = torch.where(drawer_dof_pos > 0.2, rewards + 10, rewards)
    rewards = torch.where(drawer_dof_pos > 0.35, rewards + task_complete_bonus, rewards)

    successes = torch.where(
        drawer_dof_pos > 0.35,
        torch.ones_like(drawer_dof_pos),
        successes,
    )

    return rewards, successes
