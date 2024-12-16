# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
)

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class OpenPickPlaceSceneCfg(InteractiveSceneCfg):
    """configuration for a franka pandas pick-and-place scene"""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.675)),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # robot
    robot: ArticulationCfg = MISSING

    # ee_frame
    ee_frame: FrameTransformerCfg = MISSING

    # lid_frame
    lid_frame: FrameTransformerCfg = MISSING

    # table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/my_usd/table/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, -0.04)),
    )

    # object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
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
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, -0.3, 0.055), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # plate
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
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
        prim_path="{ENV_REGEX_NS}/Lid",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/my_usd/bowl/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.3, 0.0)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orentation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        plate_position = ObsTerm(func=mdp.plate_position_in_robot_root_frame)
        lid_position = ObsTerm(func=mdp.lid_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (0.1, 0.2), "y": (-0.08, -0.15), "z": (0.055, 0.055)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )

    # reset_lid_position = EventTerm(
    #     func=mdp.reset_lid_position,
    #     mode="reset",
    # )

    # reset_plate_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (0.15, 0.25), "y": (0.1, 0.15), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("plate"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_lid = RewTerm(
        func=mdp.lid_ee_distance, params={"std": 0.1, "minimal_dist": 0.2}, weight=1.0
    )

    lifting_lid = RewTerm(
        func=mdp.lid_is_lifted, params={"minimal_height": 0.05}, weight=0.5
    )

    moving_lid = RewTerm(
        func=mdp.lid_is_moved,
        params={"minimal_height": 0.05, "minimal_dist": 0.2},
        weight=1.0,
    )

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "minimal_dist": 0.2},
        weight=2.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.08}, weight=5.0
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.08},
        weight=10.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.08},
        weight=5.0,
    )

    task_complete = RewTerm(
        func=mdp.task_complete_bonus,
        weight=1.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    task_complete = DoneTerm(
        func=mdp.object_reached_goal,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class OpenPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: OpenPickPlaceSceneCfg = OpenPickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
