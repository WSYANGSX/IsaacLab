from __future__ import annotations

# import relative module
import torch
import numpy as np

from gymnasium import spaces

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, FrameTransformer
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.assets import (
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    Articulation,
    ArticulationCfg,
)

# import relative func
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, sample_uniform

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

# modify default config
marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"

torch.set_printoptions(profile="full")


@configclass
class PickAndPlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10  # 500 timesteps
    decimation = 4
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,))  # 7 joint actions & 1 binary actions
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(43,))
    state_space = 0
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
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.675)),
    )

    # table
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/my_usd/table/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, -0.04), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # cube
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.3, 0.024), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # plate
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/my_usd/plate/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.2, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # ee_frame
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

    # action rate
    action_scale = 0.5

    # reward weight & threshold
    # weight
    reaching_cube_reward_weight = 1.0
    reaching_target_reward_weight = 10.0
    action_penalty_weight = -0.0001
    robot_dof_vel_penalty_weight = -0.0001

    # bonus
    cube_lifted_bonus = 5.0
    cube_in_plate_bonus = 10.0
    task_complete_bonus = 50.0

    # threshold
    cube_lifted_height = 0.08
    ee_cube_dist_std = 0.1
    cube_plate_dist_std = 0.1
    plate_radius = 0.1815
    fall_height = -0.05


class PickAndPlaceEnv(DirectRLEnv):
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

    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.actions_low = self.cfg.action_space.low[0]
        self.actions_high = self.cfg.action_space.high[0]

        # robot propertities
        self.arm_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        self.gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
        self.arm_entity_cfg.resolve(self.scene)
        self.gripper_entity_cfg.resolve(self.scene)

        self.arm_joint_idx = self.arm_entity_cfg.joint_ids  # type: ignore
        self.gripper_joint_idx = self.gripper_entity_cfg.joint_ids  # type: ignore

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.arm_offset = self._robot.data.default_joint_pos[:, self.arm_joint_idx].clone()

        self.gripper_open_command = self.robot_dof_upper_limits[self.gripper_joint_idx].clone().repeat(self.num_envs, 1)
        self.gripper_close_command = (
            self.robot_dof_lower_limits[self.gripper_joint_idx].clone().repeat(self.num_envs, 1)
        )

        # buffers
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros(
            (self.num_envs, *self.cfg.action_space.shape),  # type: ignore
            dtype=torch.float32,
            device=self.device,
        )
        self.pre_actions = torch.zeros_like(self.actions)

        # flags
        self.cube_lifted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.cube_in_plate = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # successes tracker
        self.successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        print("[INFO] *************** Task load complete ***************")

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

        # ee_frame & lid_frame
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
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)  # type: ignore

        # reset robot state
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

        # plate state (0.2~0.25, 0.35, 0)
        plate_pos = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)  # type: ignore
        plate_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)  # type: ignore
        _x = torch.rand(len(env_ids), device=self.device) * 0.1 + 0.15  # type: ignore
        _y = torch.rand(len(env_ids), device=self.device) * 0.1 + 0.15  # type: ignore
        plate_pos[:, 0], plate_pos[:, 1] = _x, _y
        self._plate.write_root_pose_to_sim(
            torch.cat((plate_pos + self.scene.env_origins[env_ids], plate_rot), dim=-1),
            env_ids=env_ids,  # type: ignore
        )

        # cube state
        cube_pos = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)  # type: ignore
        cube_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)  # type: ignore
        _x = torch.rand(len(env_ids), device=self.device) * 0.1 + 0.15  # type: ignore
        _y = torch.rand(len(env_ids), device=self.device) * (-0.15) - 0.15  # type: ignore
        cube_pos[:, 0], cube_pos[:, 1], cube_pos[:, 2] = _x, _y, 0.055
        self._cube.write_root_pose_to_sim(
            torch.cat((cube_pos + self.scene.env_origins[env_ids], cube_rot), dim=-1),
            env_ids=env_ids,  # type: ignore
        )

        # compute physx data
        self.sim.step(render=False)
        self.scene.update(self.cfg.sim.dt)

        # need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()  # type: ignore

        self.pre_actions[env_ids] = torch.zeros_like(self.pre_actions[env_ids])
        self.actions[env_ids] = torch.zeros_like(self.actions[env_ids])
        self.cube_lifted[env_ids] = 0
        self.cube_in_plate[env_ids] = 0

        self.successes[env_ids] = 0

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.pre_actions[:] = self.actions[:]
        self.actions[:] = actions.clone().clamp(self.actions_low, self.actions_high)[:]

        # arm action
        arm_actions = self.actions[:, :7]
        arm_targets = self.arm_offset + self.cfg.action_scale * arm_actions

        # gripper action
        gripper_actions = self.actions[:, 7].clone().view(-1, 1).repeat(1, 2)
        gripper_targets = torch.where(gripper_actions > 0, self.gripper_open_command, self.gripper_close_command)

        self.robot_dof_targets = torch.clamp(
            torch.cat((arm_targets, gripper_targets), dim=-1),
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when time out
        truncated = (
            self.episode_length_buf >= self.max_episode_length - 1
        )  # 判断符自动将tensor转换为bool类型，& |按位操作

        # reset when cube fall or out of reach
        cube_fall = self.cube_pos_b[:, 2] <= self.cfg.fall_height

        # task complete (do not apply)
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        return terminated, truncated | cube_fall

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot_dof_pos,
                self.robot_dof_vel,
                self.ee_pos_b,
                self.ee_rot_b,
                self.cube_pos_b,
                self.cube_rot_b,
                self.plate_pos_b,
                self.actions,
            ),
            dim=-1,
        )

        observations = {"policy": obs}

        if self.cfg.asymmetric_obs:
            states = self._get_states()
            observations = {"policy": obs, "critic": states}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward, self.successes = compute_rewards(
            self.ee_pos_b,
            self.cube_pos_b,
            self.plate_pos_b,
            self.cfg.ee_cube_dist_std,
            self.cfg.cube_plate_dist_std,
            self.actions,
            self.pre_actions,
            self.robot_dof_vel,
            self.cfg.reaching_cube_reward_weight,
            self.cfg.reaching_target_reward_weight,
            self.cube_lifted,
            self.cube_in_plate,
            self.cfg.action_penalty_weight,
            self.cfg.robot_dof_vel_penalty_weight,
            self.cfg.cube_lifted_bonus,
            self.cfg.cube_in_plate_bonus,
            self.cfg.task_complete_bonus,
            self.successes,
        )

        return total_reward

    def _compute_intermediate_values(self):
        # env states
        self.robot_dof_pos = self._robot.data.joint_pos
        self.robot_dof_vel = self._robot.data.joint_vel
        self.cube_pos_b, self.cube_rot_b = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._cube.data.root_pos_w,
            self._cube.data.root_quat_w,
        )
        self.plate_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._plate.data.root_pos_w,
            self._plate.data.root_quat_w,
        )
        self.ee_pos_b, self.ee_rot_b = (
            self._ee_frame.data.target_pos_source[:, 0, :],
            self._ee_frame.data.target_quat_source[:, 0, :],
        )

        # flags
        self.cube_lifted = torch.where(
            self.cube_pos_b[:, 2] > self.cfg.cube_lifted_height,
            torch.ones_like(self.cube_lifted),
            self.cube_lifted,
        )

        self.cube_in_plate = torch.where(
            (torch.norm(self.cube_pos_b[:, :2] - self.plate_pos_b[:, :2], p=2, dim=-1) < self.cfg.plate_radius)
            & (self.cube_pos_b[:, 2] < 0.03),
            torch.ones_like(self.cube_in_plate),
            self.cube_in_plate,
        )


@torch.jit.script
def compute_rewards(
    ee_pos_b: torch.Tensor,
    cube_pos_b: torch.Tensor,
    plate_pos_b: torch.Tensor,
    ee_cube_dist_std: float,
    cube_plate_dist_std: float,
    actions: torch.Tensor,
    pre_actions: torch.Tensor,
    robot_dof_vel: torch.Tensor,
    reaching_cube_reward_weight: float,
    reaching_target_reward_weight: float,
    cube_lifted: torch.Tensor,
    cube_in_plate: torch.Tensor,
    action_penalty_weight: float,
    robot_dof_vel_penalty_weight: float,
    cube_lifted_bonus: float,
    cube_in_plate_bonus: float,
    task_complete_bonus: float,
    successes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # stage1: approach to cube, lift cube
    ee_cube_dist = torch.norm(ee_pos_b - cube_pos_b, p=2, dim=-1)
    reaching_cube_reward = (1 - torch.tanh(ee_cube_dist / ee_cube_dist_std)) * reaching_cube_reward_weight
    # print("reaching_cube_reward: ", reaching_cube_reward)

    cube_lifted_reward = cube_lifted * cube_lifted_bonus
    # print("cube_lifted_reward: ", cube_lifted_reward)

    # stage2: pick cube to plate, place cube
    cube_plate_dist = torch.norm(plate_pos_b - cube_pos_b, p=2, dim=-1)
    reaching_target_reward = (
        (1 - torch.tanh(cube_plate_dist / cube_plate_dist_std)) * cube_lifted * reaching_target_reward_weight
    )
    # print("reaching_target_reward:", reaching_target_reward)

    # cube_in_plate_reward = cube_lifted * cube_lifted * cube_in_plate_bonus
    # print("cube_in_plate_reward: ", cube_in_plate_reward)

    # # stage3: place cube
    # leaving_cube_reward = (
    #     torch.tanh(ee_cube_dist / ee_cube_std) * cube_lifted * cube_in_plate * leaving_cube_reward_weight
    # )
    # # print("leaving_cube_reward: ", leaving_cube_reward)

    # action penalty
    action_penalty = torch.sum((actions - pre_actions) ** 2, dim=-1) * action_penalty_weight
    # print("action_penalty:", action_penalty)

    # vel penalty
    robot_dof_vel_penalty = torch.sum(robot_dof_vel**2, dim=-1) * robot_dof_vel_penalty_weight

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = (
        reaching_cube_reward
        + cube_lifted_reward
        + reaching_target_reward
        # + cube_in_plate_reward
        # + leaving_cube_reward
        + action_penalty
        + robot_dof_vel_penalty
    )

    # task complete
    ee_succ = ee_pos_b[:, 2] > 0.08
    successes = torch.where(cube_in_plate & ee_succ, torch.ones_like(successes), successes)

    reward = torch.where(successes, reward + task_complete_bonus, reward)

    return reward, successes
