# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------
# File Name:        pick_and_place_env
# Author:           Xiaofan Yang
# Version:          0.1
# Created:          2024/11/18
# Description:      Main Function:    environment of pick and place task

# History:
#       <author>             <version>       <time>      <desc>
#       Xiaofan Yang         0.1             2024/11/18  xxx
# ------------------------------------------------------------------

from __future__ import annotations

import torch


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class PickAndPlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0  # 250 timesteps
    decimation = 2
    action_space: int = 8
    observation_space: int = 43
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/my_usd/table/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, -0.04)),
    )

    # cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.3, 0.055), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # plate
    plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/my_usd/plate/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.2, 0.0)),
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.675)),
        collision_group=-1,
    )

    # ee frame
    marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
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

    action_scale = 0.5

    # reward weight & threshold
    ee_cube_dist_std = 0.1
    cube_plate_dist_std = 0.3
    cube_lifted_height = 0.08


class PickAndPlaceEnv(DirectRLEnv):
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

    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # robot propertities
        self.arm_joint_ids, _ = self._robot.find_joints(name_keys=["panda_joint.*"])
        self.arm_offset = self._robot.data.default_joint_pos[:, self.arm_joint_ids].clone()

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.gripper_open_command = self.robot_dof_upper_limits[-1]
        self.gripper_close_command = self.robot_dof_lower_limits[-1]

        # buffers
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.actions = torch.zeros(size=(self.num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros_like(self.actions)

        # success rate
        self.successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _setup_scene(self):
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # table
        self._table = RigidObject(self.cfg.table)
        self.scene.rigid_objects["table"] = self._table

        # cube
        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        # plate
        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        # ee_frame
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.scene.sensors["ee_frame"] = self._ee_frame

        # ground
        self._ground = self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
            orientation=self.cfg.ground.init_state.rot,
        )

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.ground.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions[:] = actions

        arm_targets = self.arm_offset + self.actions[:, :7] * self.cfg.action_scale
        gripper_targets = (
            torch.where(self.actions[:, 7] >= 0, self.gripper_open_command, self.gripper_close_command)
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
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        cube_fall = self._cube.data.root_pos_w[:, 2] < -0.05
        return terminated, truncated | cube_fall

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        reward, self.successes = self._compute_rewards(
            self.actions,
            self.prev_actions,
            self.ee_pos_l,
            self.cube_pos_l,
            self.plate_pos_l,
            self._robot.data.joint_vel,
            self.cfg.ee_cube_dist_std,
            self.cfg.cube_plate_dist_std,
            self.cfg.cube_lifted_height,
            self.successes,
        )

        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # plate
        plate_pos = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)  # type: ignore
        plate_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)  # type: ignore
        _x = torch.rand(len(env_ids), device=self.device) * 0.1 + 0.2  # type: ignore
        _y = torch.rand(len(env_ids), device=self.device) * 0.1 + 0.2  # type: ignore
        plate_pos[:, 0], plate_pos[:, 1] = _x, _y
        self._plate.write_root_pose_to_sim(
            torch.cat((plate_pos + self.scene.env_origins[env_ids], plate_rot), dim=-1),
            env_ids=env_ids,  # type: ignore
        )

        # cube
        cube_pos = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)  # type: ignore
        cube_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)  # type: ignore
        _x = torch.rand(len(env_ids), device=self.device) * 0.1 + 0.15  # type: ignore
        _y = torch.rand(len(env_ids), device=self.device) * (-0.15) - 0.15  # type: ignore
        cube_pos[:, 0], cube_pos[:, 1], cube_pos[:, 2] = _x, _y, 0.055
        self._cube.write_root_pose_to_sim(
            torch.cat((cube_pos + self.scene.env_origins[env_ids], cube_rot), dim=-1),
            env_ids=env_ids,  # type: ignore
        )

        # TODO:compute physx data
        self.sim.step(render=False)
        self.scene.update(1e-10)

        # compute intermediate state values after reset
        self._compute_intermediate_values(env_ids)

        self.actions[env_ids] = torch.zeros_like(self.actions[env_ids])
        self.prev_actions[env_ids] = torch.zeros_like(self.prev_actions[env_ids])

        self.successes[env_ids] = 0

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._robot.data.joint_pos,
                self._robot.data.joint_vel,
                self.cube_pos_l,
                self.cube_quat_l,
                self.ee_pos_l,
                self.ee_quat_l,
                self.plate_pos_l,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.ee_pos_l, self.ee_quat_l = (
            self._ee_frame.data.source_pos_w - self.scene.env_origins,
            self._ee_frame.data.source_quat_w,
        )
        self.cube_pos_l, self.cube_quat_l = (
            self._cube.data.root_pos_w - self.scene.env_origins,
            self._cube.data.root_quat_w,
        )
        self.plate_pos_l, self.plate_quat_l = (
            self._plate.data.root_pos_w - self.scene.env_origins,
            self._plate.data.root_quat_w,
        )

    def _compute_rewards(
        self,
        actions,
        prev_actions,
        ee_pos_l,
        cube_pos_l,
        plate_pos_l,
        robot_dof_vel,
        ee_cube_dist_std,
        cube_plate_dist_std,
        cube_lifted_height,
        successes,
    ):
        # distance from ee to the cube
        ee_cube_dist = torch.norm(ee_pos_l - cube_pos_l, p=2, dim=-1)
        reaching_cube_reward = 1 - torch.tanh(ee_cube_dist / ee_cube_dist_std)

        # cube lifted reward
        cube_lifted_reward = torch.where(cube_pos_l[:, 2] > cube_lifted_height, 1.0, 0.0)

        # distance from cube to target
        target_cube_xy_dist = torch.norm(plate_pos_l[:, :2] - cube_pos_l[:, :2], dim=-1)
        reaching_target_reward = (cube_pos_l[:, 2] > cube_lifted_height) * (
            1 - torch.tanh(target_cube_xy_dist / cube_plate_dist_std)
        )

        # regularization on the actions
        action_penalty = torch.sum(torch.square(actions - prev_actions), dim=-1)

        # regularization on the joint vels
        dof_vel_penalty = torch.sum(torch.square(robot_dof_vel), dim=-1)

        rewards = (
            reaching_cube_reward * 1 * self.step_dt
            + cube_lifted_reward * 5 * self.step_dt
            + reaching_target_reward * 16 * self.step_dt
            + action_penalty * (-1e-4) * self.step_dt
            + dof_vel_penalty * (-1e-4) * self.step_dt
        )

        # bonus for task
        task_complete = torch.where(
            (target_cube_xy_dist <= 0.15) & (ee_pos_l[:, 2] >= 0.1) & (cube_pos_l[:, 2] <= 0.055),
            1,
            0,
        )
        rewards = torch.where(task_complete == 1, rewards + 50 * self.step_dt, rewards)

        successes = torch.where(
            task_complete == 1,
            torch.ones_like(successes),
            torch.zeros_like(successes),
        )

        return rewards, successes
