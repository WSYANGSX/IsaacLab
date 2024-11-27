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


@configclass
class OpenPickPlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5  # 720 timesteps
    decimation = 2
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,))  # 7 joint actions & 1 binary actions
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(46,))
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.42, 0.17)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, -0.04)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.2, 0.0)),
    )

    # lid
    lid = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Lid",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/my_usd/bowl/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.3, 0.0)),
    )

    # ee_frame
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
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

    # lid_frame
    lid_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Lid/lid",
                name="lid",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.123),
                ),
            ),
        ],
    )

    # action rate
    action_scale = 0.5

    # reward weight & threshold
    # weight
    reaching_lid_reward_weight = 2.0
    reaching_cube_reward_weight = 3.0
    lid_lifted_reward_weight = 1.0
    lid_moving_reward_weight = 1.0
    cube_lifted_reward_weight = 10.0
    cube_moving_reward_weight = 15.0
    action_penalty_weight = -0.001

    # bonus
    lid_lifted_bonus = 1.0
    cube_lifted_bonus = 1.0
    task_complete_bonus = 500

    # threshold
    cube_in_plate_dist = 0.015
    lid_lifted_height = 0.04
    cube_lifted_height = 0.04
    lid_moved_dist = 0.3
    ee_lid_std = 0.2
    ee_cube_std = 0.2
    cube_plate_std = 0.2


class OpenPickPlaceEnv(DirectRLEnv):
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

    cfg: OpenPickPlaceEnvCfg

    def __init__(self, cfg: OpenPickPlaceEnvCfg, render_mode: str | None = None, **kwargs) -> None:
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

        self.gripper_open_command = self.robot_dof_upper_limits[self.gripper_joint_idx].clone()
        self.gripper_close_command = self.robot_dof_lower_limits[self.gripper_joint_idx].clone()

        # buffers
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.actions = torch.zeros((self.num_envs, *self.cfg.action_space.shape), device=self.device)  # type: ignore

        # flags
        self.lid_lifted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.lid_moved = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.cube_lifted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # successes tracker
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
        self.scene.rigid_objects["plate"] = self._cube

        # lid
        self._lid = RigidObject(self.cfg.lid)
        self.scene.rigid_objects["lid"] = self._lid

        # ee_frame & lid_frame
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self._lid_frame = FrameTransformer(self.cfg.lid_frame)
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.sensors["lid_frame"] = self._lid_frame

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

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(self.actions_low, self.actions_high)

        # arm action
        arm_actions = self.actions[:, :7]
        arm_targets = self.arm_offset + self.cfg.action_scale * arm_actions

        # gripper action
        gripper_actions = self.actions[:, 7].clone().view(-1, 1).repeat(1, 2)
        gripper_targets = torch.where(gripper_actions >= 0, self.gripper_open_command, self.gripper_close_command)

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
        cube_fall = self.cube_pos_b[:, 2] <= 0

        # task complete
        terminated = torch.norm(self.cube_pos_b - self.plate_pos_b, p=2, dim=-1) <= self.cfg.cube_in_plate_dist

        return terminated, truncated | cube_fall

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
        _x = torch.rand(len(env_ids), device=self.device) * 0.05 + 0.2  # type: ignore
        plate_pos[:, 0], plate_pos[:, 1] = _x, 0.35
        self._plate.write_root_pose_to_sim(
            torch.cat((plate_pos + self.scene.env_origins[env_ids], plate_rot), dim=-1),
            env_ids=env_ids,  # type: ignore
        )

        # cube and lid state (0.2~0.25, -0.2~-0.35, 0.05)
        cube_pos = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)  # type: ignore
        cube_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)  # type: ignore
        _x = torch.rand(len(env_ids), device=self.device) * 0.05 + 0.2  # type: ignore
        _y = torch.rand(len(env_ids), device=self.device) * -0.15 - 0.2  # type: ignore
        cube_pos[:, 0], cube_pos[:, 1], cube_pos[:, 2] = _x, _y, 0.05
        self._cube.write_root_pose_to_sim(
            torch.cat((cube_pos + self.scene.env_origins[env_ids], cube_rot), dim=-1),
            env_ids=env_ids,  # type: ignore
        )

        lid_pos, lid_rot = cube_pos.clone(), cube_rot.clone()
        lid_pos[:, 2] = 0.0
        self._lid.write_root_pose_to_sim(
            torch.cat((lid_pos + self.scene.env_origins[env_ids], lid_rot), dim=-1),
            env_ids=env_ids,  # type: ignore
        )

        # compute physx data
        self.sim.step(render=False)
        self.scene.update(self.cfg.sim.dt)

        # need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()  # type: ignore

        self.successes[env_ids] = 0

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot_dof_pos,
                self.robot_dof_vel,
                self.cube_pos_b,
                self.cube_rot_b,
                self.lid_pos_b,
                self.lid_rot_b,
                self.plate_pos_b,
                self.lid_lifted.view(-1, 1),
                self.lid_moved.view(-1, 1),
                self.cube_lifted.view(-1, 1),
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
            self.episode_length_buf,
            self.successes,
            self.ee_pos_b,
            self.cube_pos_b,
            self.lid_pos_b,
            self.handle_pos_b,
            self.plate_pos_b,
            self.lid_lifted,
            self.lid_moved,
            self.cube_lifted,
            self.cfg.ee_lid_std,
            self.cfg.ee_cube_std,
            self.cfg.cube_plate_std,
            self.cfg.reaching_lid_reward_weight,
            self.cfg.reaching_cube_reward_weight,
            self.cfg.lid_lifted_reward_weight,
            self.cfg.lid_moving_reward_weight,
            self.cfg.cube_lifted_reward_weight,
            self.cfg.cube_moving_reward_weight,
            self.cfg.action_penalty_weight,
            self.cfg.lid_lifted_bonus,
            self.cfg.cube_lifted_bonus,
            self.cfg.task_complete_bonus,
            self.cfg.cube_in_plate_dist,
            self.actions,
        )

        return total_reward

    def _compute_intermediate_values(self):
        self.robot_dof_pos = self._robot.data.joint_pos
        self.robot_dof_vel = self._robot.data.joint_vel
        self.cube_pos_b, self.cube_rot_b = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._cube.data.root_pos_w,
            self._cube.data.root_quat_w,
        )
        self.lid_pos_b, self.lid_rot_b = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._lid.data.root_pos_w,
            self._lid.data.root_quat_w,
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
        self.handle_pos_b, self.handle_rot_b = (
            self._lid_frame.data.target_pos_source[:, 0, :],
            self._lid_frame.data.target_quat_source[:, 0, :],
        )

        self.lid_lifted = torch.where(
            self.lid_pos_b[:, 2] > self.cfg.lid_lifted_height,
            torch.ones_like(self.lid_lifted),
            torch.zeros_like(self.lid_lifted),
        )

        self.lid_moved = torch.where(
            torch.norm(self.cube_pos_b[:, :2] - self.lid_pos_b[:, :2], dim=-1, p=2) > self.cfg.lid_moved_dist,
            torch.ones_like(self.lid_moved),
            torch.zeros_like(self.lid_moved),
        )

        self.cube_lifted = torch.where(
            self.cube_pos_b[:, 2] > self.cfg.cube_lifted_height,
            torch.ones_like(self.cube_lifted),
            torch.zeros_like(self.cube_lifted),
        )


@torch.jit.script
def compute_rewards(
    eposide_length_buf: torch.Tensor,
    successes: torch.Tensor,
    ee_pos_b: torch.Tensor,
    cube_pos_b: torch.Tensor,
    lid_pos_b: torch.Tensor,
    handle_pos_b: torch.Tensor,
    plate_pos_b: torch.Tensor,
    lid_lifted: torch.Tensor,
    lid_moved: torch.Tensor,
    cube_lifted: torch.Tensor,
    ee_lid_std: float,
    ee_cube_std: float,
    cube_plate_std: float,
    reaching_lid_reward_weight: float,
    reaching_cube_reward_weight: float,
    lid_lifted_reward_weight: float,
    lid_moving_reward_weight: float,
    cube_lifted_reward_weight: float,
    cube_moving_reward_weight: float,
    action_penalty_weight: float,
    lid_lifted_bonus: float,
    cube_lifted_bonus: float,
    task_complete_bonus: float,
    cube_in_plate_dist: float,
    actions: torch.Tensor,
):
    # stage1: approach to lid, lift lid, move lid
    # reaching reward
    ee_lid_dist = torch.norm(ee_pos_b - handle_pos_b, p=2, dim=-1)
    reaching_lid_reward = (
        (1 - torch.tanh(ee_lid_dist / ee_lid_std)) * lid_moved.logical_not() * reaching_lid_reward_weight
    )
    # print("reaching_lid_reward:", reaching_lid_reward)

    # lifting reward
    lid_lifted_reward = -1 * lid_lifted.logical_not() * lid_moved.logical_not() * lid_lifted_reward_weight
    # print("lid_lifted_reward:", lid_lifted_reward)

    # moving reward
    lid_moveing_reward = -1 * lid_moved.logical_not() * lid_moving_reward_weight
    # print("lid_moveing_reward:", lid_moveing_reward)

    # stage2: approach to cube, lift cube, move cube
    stage2_reward = 2 * lid_moved

    # reaching reward
    ee_cube_dist = torch.norm(ee_pos_b - cube_pos_b, p=2, dim=-1)
    reaching_cube_reward = (2 - torch.tanh(ee_cube_dist / ee_cube_std)) * lid_moved * reaching_cube_reward_weight
    # print("reaching_cube_reward:", reaching_cube_reward)

    # lift reward
    cube_lifted_reward = -1 * lid_moved * cube_lifted.logical_not() * cube_lifted_reward_weight
    # print("cube_lifted_reward:", cube_lifted_reward)

    # moving cube
    cube_plate_dist = torch.norm(cube_pos_b - plate_pos_b, p=2, dim=-1)
    cube_moving_reward = (
        (1 - torch.tanh(cube_plate_dist / cube_plate_std)) * lid_moved * cube_lifted * cube_moving_reward_weight
    )
    # print("cube_moving_reward:", cube_moving_reward)

    # action penalty
    action_penalty = torch.sum(actions**2, dim=-1) * action_penalty_weight
    # print("action_penalty:", action_penalty)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = (
        reaching_lid_reward
        + lid_lifted_reward
        + lid_moveing_reward
        + stage2_reward
        + reaching_cube_reward
        + cube_lifted_reward
        + cube_moving_reward
        + action_penalty
    )  # reward的shape为(num_envs, )

    # Success bonus: cube is within plate
    task_complete = cube_plate_dist <= cube_in_plate_dist
    successes += task_complete

    reward = torch.where(task_complete == 1, reward + task_complete_bonus, reward)

    return reward, successes
