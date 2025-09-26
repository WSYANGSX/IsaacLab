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
from typing import Literal

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg

from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots import FRANKA_PANDA_CFG  # isort: skip


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
        dt=1 / 100,
        render_interval=decimation,
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
            usd_path="/home/yangxf/MyProjects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/pick_and_place/usd/table/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.42, 0.17)),
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
            usd_path="/home/yangxf/MyProjects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/pick_and_place/usd/plate/parts/Part_1_JHD.usd",
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

    train_method: Literal["ppo-dense", "ppo-sparse", "ddpg-dense", "ddpg-sparse"] = "ddpg-dense"

    # reward weight
    reaching_cube_reward_weight = 1
    cube_lifted_reward_weight = 5
    reaching_plate_reward_weight = 16
    action_penalty_weight = -1e-4
    dof_vel_penalty_weight = -1e-4

    # threshold
    cube_fall_height = -0.05
    cube_lifted_height = 0.15
    ee_cube_dist_std = 0.1
    cube_plate_dist_std = 0.3


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

        if self.cfg.train_method == "ppo-dense" or self.cfg.train_method == "ppo-sparse":
            self.action_scale = 0.5
        elif self.cfg.train_method == "ddpg-dense" or self.cfg.train_method == "ddpg-sparse":
            self.action_scale = 0.5
            self.cfg.cube_lifted_height = 0.08
        else:
            raise ValueError("Undefinded train method.")

        # robot properties
        self.arm_joint_ids, _ = self._robot.find_joints(name_keys=["panda_joint.*"])
        self.arm_offset = self._robot.data.default_joint_pos[:, self.arm_joint_ids].clone()

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.gripper_open_command = self.robot_dof_upper_limits[-1]
        self.gripper_close_command = self.robot_dof_lower_limits[-1]

        # buffers
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros_like(self.actions)

        # flags
        self.cube_lifted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.curriculum_performed = False

        # success rate
        self.successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        print("[INFO]: Train method: ", self.cfg.train_method)
        print("[INFO]: Action scale: ", self.action_scale)
        print("[INFO]: Cube lifted height: ", self.cfg.cube_lifted_height)

    def _setup_scene(self):
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # table
        self._table = RigidObject(self.cfg.table)
        self._fix_rigid_body_api(self._table)
        self.scene.rigid_objects["table"] = self._table

        # cube
        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        # plate
        self._plate = RigidObject(self.cfg.plate)
        self._fix_rigid_body_api(self._plate)
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

    def _fix_rigid_body_api(self, asset):
        """fix RigidBodyAPI"""
        from pxr import UsdPhysics

        # get prim
        prim_path = asset.cfg.prim_path.replace(".*", "0")  # get first prim path
        stage = self.sim.stage
        prim = stage.GetPrimAtPath(prim_path)

        if prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"Applying RigidBodyAPI to {prim_path}")
            UsdPhysics.RigidBodyAPI.Apply(prim)

            # apply CollisionAPI
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions[:] = actions

        arm_targets = self.arm_offset + self.actions[:, :7] * self.action_scale
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
        cube_fall = self._cube.data.root_pos_w[:, 2] < self.cfg.cube_fall_height
        return terminated, truncated | cube_fall

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        if self.cfg.train_method == "ppo-dense" or self.cfg.train_method == "ddpg-dense":
            reward, self.successes = compute_dense_rewards(
                self.actions,
                self.prev_actions,
                self.ee_pos_l,
                self.cube_pos_l,
                self.plate_pos_l,
                self._robot.data.joint_vel,
                self.cfg.ee_cube_dist_std,
                self.cfg.cube_plate_dist_std,
                self.cfg.cube_lifted_height,
                self.cfg.cube_fall_height,
                self.cfg.reaching_cube_reward_weight,
                self.cfg.cube_lifted_reward_weight,
                self.cfg.reaching_plate_reward_weight,
                self.cfg.action_penalty_weight,
                self.cfg.dof_vel_penalty_weight,
                self.cube_lifted,
                self.successes,
            )
        else:
            reward, self.successes = compute_sparse_rewards(
                self.actions,
                self.prev_actions,
                self.ee_pos_l,
                self.cube_pos_l,
                self.plate_pos_l,
                self._robot.data.joint_vel,
                self.cfg.cube_lifted_height,
                self.cfg.cube_fall_height,
                self.cfg.cube_lifted_reward_weight,
                self.cfg.action_penalty_weight,
                self.cfg.dof_vel_penalty_weight,
                self.cube_lifted,
                self.successes,
            )

        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)  # type: ignore

        # TODO:randomize robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)  # type: ignore
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)  # type: ignore

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

        self.actions[env_ids] = 0
        self.prev_actions[env_ids] = 0

        self.cube_lifted[env_ids] = 0
        self.successes[env_ids] = 0

        # compute intermediate state values after reset
        self._compute_intermediate_values(env_ids)

        # curriculum
        if self.cfg.train_method == "ppo-dense" or self.cfg.train_method == "ppo-sparse":
            self._curriculum()

    def _curriculum(self) -> None:
        if (self.common_step_counter >= 40000) & (self.curriculum_performed is False):
            print("****************** curriculum performed ******************")
            self.cfg.action_penalty_weight = -1e-1
            self.cfg.dof_vel_penalty_weight = -1e-1
            self.curriculum_performed = True

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
            self._ee_frame.data.target_pos_source[:, 0, :],
            self._ee_frame.data.target_quat_source[:, 0, :],
        )

        self.cube_pos_l, self.cube_quat_l = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._cube.data.root_pos_w,
            self._cube.data.root_quat_w,
        )

        self.plate_pos_l, self.plate_quat_l = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._plate.data.root_pos_w,
            self._plate.data.root_quat_w,
        )

        # flags
        self.cube_lifted = torch.where(
            self.cube_pos_l[:, 2] > self.cfg.cube_lifted_height, torch.ones_like(self.cube_lifted), self.cube_lifted
        )


@torch.jit.script
def compute_dense_rewards(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    ee_pos_l: torch.Tensor,
    cube_pos_l: torch.Tensor,
    plate_pos_l: torch.Tensor,
    robot_dof_vel: torch.Tensor,
    ee_cube_dist_std: float,
    cube_plate_dist_std: float,
    cube_lifted_height: float,
    cube_fall_height: float,
    reaching_cube_reward_weight: float,
    cube_lifted_reward_weight: float,
    reaching_plate_reward_weight: float,
    action_penalty_weight: float,
    dof_vel_penalty_weight: float,
    cube_lifted: torch.Tensor,
    successes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # distance from ee to the cube
    ee_cube_dist = torch.norm(ee_pos_l - cube_pos_l, p=2, dim=-1)
    reaching_cube_reward = 1 - torch.tanh(ee_cube_dist / ee_cube_dist_std)

    # cube lifted reward
    cube_lifted_reward = torch.where(cube_pos_l[:, 2] > cube_lifted_height, 1.0, 0.0)

    # distance from cube to target
    target_pos = plate_pos_l.clone()
    target_pos[:, 2] = 0.1
    target_cube_dist = torch.norm(target_pos - cube_pos_l, dim=-1)
    reaching_target_reward = (cube_pos_l[:, 2] > cube_lifted_height) * (
        1 - torch.tanh(target_cube_dist / cube_plate_dist_std)
    )

    # regularization on the actions
    action_penalty = torch.sum(torch.square(actions - prev_actions), dim=-1)

    # regularization on the joint velocities
    dof_vel_penalty = torch.sum(torch.square(robot_dof_vel), dim=-1)

    rewards = (
        reaching_cube_reward * reaching_cube_reward_weight
        + cube_lifted_reward * cube_lifted_reward_weight
        + reaching_target_reward * reaching_plate_reward_weight
        + action_penalty * action_penalty_weight
        + dof_vel_penalty * dof_vel_penalty_weight
    )

    # cube fall
    rewards = torch.where(cube_pos_l[:, 2] <= cube_fall_height, rewards - 1000, rewards)

    # bonus for task
    task_complete = torch.where(
        (target_cube_dist <= 0.15) & (ee_pos_l[:, 2] >= 0.1) & (cube_pos_l[:, 2] <= 0.055) & cube_lifted,
        torch.ones_like(target_cube_dist),
        torch.zeros_like(target_cube_dist),
    )
    rewards = torch.where(task_complete == 1, rewards + 50, rewards)

    successes = torch.where(
        task_complete == 1,
        torch.ones_like(successes),
        torch.zeros_like(successes),
    )

    return rewards, successes


@torch.jit.script
def compute_sparse_rewards(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    ee_pos_l: torch.Tensor,
    cube_pos_l: torch.Tensor,
    plate_pos_l: torch.Tensor,
    robot_dof_vel: torch.Tensor,
    cube_lifted_height: float,
    cube_fall_height: float,
    cube_lifted_reward_weight: float,
    action_penalty_weight: float,
    dof_vel_penalty_weight: float,
    cube_lifted: torch.Tensor,
    successes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cube lifted reward
    cube_lifted_reward = torch.where(cube_pos_l[:, 2] > cube_lifted_height, 1.0, 0.0)

    # distance from cube to target
    target_pos = plate_pos_l.clone()
    target_pos[:, 2] = 0.1
    target_cube_dist = torch.norm(target_pos - cube_pos_l, dim=-1)

    # regularization on the actions
    action_penalty = torch.sum(torch.square(actions - prev_actions), dim=-1)

    # regularization on the joint velocities
    dof_vel_penalty = torch.sum(torch.square(robot_dof_vel), dim=-1)

    rewards = (
        cube_lifted_reward * cube_lifted_reward_weight
        + action_penalty * action_penalty_weight
        + dof_vel_penalty * dof_vel_penalty_weight
    )

    # cube fall
    rewards = torch.where(cube_pos_l[:, 2] <= cube_fall_height, rewards - 1000, rewards)

    # bonus for task
    task_complete = torch.where(
        (target_cube_dist <= 0.15) & (ee_pos_l[:, 2] >= 0.1) & (cube_pos_l[:, 2] <= 0.055) & cube_lifted,
        torch.ones_like(target_cube_dist),
        torch.zeros_like(target_cube_dist),
    )
    rewards = torch.where(task_complete == 1, rewards + 50, rewards)

    successes = torch.where(
        task_complete == 1,
        torch.ones_like(successes),
        torch.zeros_like(successes),
    )

    return rewards, successes
