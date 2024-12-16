from __future__ import annotations

import torch
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, OffsetCfg
from omni.isaac.lab.markers import FRAME_MARKER_CFG
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewardTerm
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from my_projects.utils.path import get_logs_path

import omni.isaac.lab_tasks.manager_based.GBAGC_RL.pick_and_place.mdp as mdp


torch.set_printoptions(profile="full")


###
# scene define
###
@configclass
class PickAndPlaceSceneCfg(InteractiveSceneCfg):
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
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # ee_frame
    ee_frame: FrameTransformerCfg = MISSING

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
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
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


###
# MDP settings
###

# get low policy action policy
logs_path = get_logs_path()

# get subgoals_list
subgoals_list = [
    [0.25, -0.3, 0.25, 0.0, 1.0, 0.0, 0.0],
    [0.25, -0.3, 0.04, 0.0, 1.0, 0.0, 0.0],
    [0.25, 0.2, 0.15, 0.0, 1.0, 0.0, 0.0],
]


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    subgoals = mdp.SubgoalsCommandCfg(
        resampling_time_range=(0, 0),
        subgoals_list=subgoals_list,
        asset_name="robot",
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.PreTrainedArmActionCfg(
        policy_path=f"{logs_path}/rl_games/franka_prtpr_jointspace_direct/2024-09-08_23-16-09/",  # create this path after test
        mode="precision",
    )
    gripper_action: ActionTerm = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        cube_pose = ObsTerm(
            func=mdp.get_asset_local_pose,
            params={"asset_cfg": SceneEntityCfg("cube")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        ee_pose = ObsTerm(
            func=mdp.get_ee_local_pose,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        ee_cube_dist = ObsTerm(
            func=mdp.get_ee_cube_dist,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "cube_cfg": SceneEntityCfg("cube"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        plate_pose = ObsTerm(
            func=mdp.get_asset_local_pose,
            params={"asset_cfg": SceneEntityCfg("plate")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        subgoal_pose = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "subgoals"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        gripper_joint_pos = ObsTerm(
            func=mdp.get_gripper_position,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["panda_finger_joint.*"]
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        gripper_joint_vel = ObsTerm(
            func=mdp.get_gripper_velocity,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["panda_finger_joint.*"]
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # subgoal reach
    subgoal_reach = RewardTerm(
        func=mdp.subgoal_reach,
        weight=1,
        params={
            "pos_threshold": 0.01,
            "rot_threshold": 0.1,
            "subgoal_reach_bonus": 100,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "subgoal_cmd_name": "subgoals",
        },
    )

    # task complete
    final_goal_reach = RewardTerm(
        func=mdp.task_goal_reach,
        weight=1,
        params={
            "pos_threshold": 0.05,
            "final_goal_reach_bonus": 500,
            "cube_cfg": SceneEntityCfg("cube"),
            "plate_cfg": SceneEntityCfg("plate"),
        },
    )

    eposide_length = RewardTerm(
        func=mdp.eposide_length,
        weight=0.001,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # task complete
    task_complete = DoneTerm(
        mdp.task_complete,
        params={
            "pos_threshold": 0.08,
            "cube_cfg": SceneEntityCfg("cube"),
            "plate_cfg": SceneEntityCfg("plate"),
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class PickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for pick and place env."""

    # Scene settings
    scene: PickAndPlaceSceneCfg = PickAndPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic setting
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP setting
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.09],
                    ),
                ),
            ],
        )
