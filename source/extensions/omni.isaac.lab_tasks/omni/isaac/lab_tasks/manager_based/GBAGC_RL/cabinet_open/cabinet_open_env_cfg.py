from __future__ import annotations

import torch
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg
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

import omni.isaac.lab_tasks.manager_based.GBAGC_RL.cabinet_open.mdp as mdp


torch.set_printoptions(profile="full")


###
# scene define
###
@configclass
class CabinetOpenSceneCfg(InteractiveSceneCfg):
    """configuration for a franka pandas cabinet-open scene"""

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

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
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

    # ee_frame
    ee_frame: FrameTransformerCfg = MISSING

    # cabinet handel_frame
    handle_frame: FrameTransformerCfg = MISSING


###
# MDP settings
###

# get low policy action policy
logs_path = get_logs_path()

# get subgoals_list
subgoals_list = [
    [0.55, 0.0, 0.7272, 0.5000, 0.5000, 0.5000, 0.5000],
    [0.6435, 0.0, 0.7272, 0.5000, 0.5000, 0.5000, 0.5000],
    [0.3, 0.0, 0.7272, 0.5000, 0.5000, 0.5000, 0.5000],
]


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    subgoals = mdp.SubgoalsCommandCfg(
        resampling_time_range=(0, 0),
        subgoals_list=subgoals_list,
        asset_name="robot",
        debug_vis=True,
        pos_threshold=0.01,
        rot_threshold=0.15,
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

        cabinet_handle_pose = ObsTerm(
            func=mdp.get_handle_local_pose,
            params={"handle_frame_cfg": SceneEntityCfg("handle_frame")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        ee_pose = ObsTerm(
            func=mdp.get_ee_local_pose,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        cabinet_ee_dist = ObsTerm(
            func=mdp.get_ee_handle_dist,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("handle_frame"),
            },
        )

        drawer_pos = ObsTerm(
            func=mdp.get_drawer_position,
            params={
                "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])
            },
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
            "pos_threshold": 0.008,
            "rot_threshold": 0.2,
            "subgoal_reach_bonus": 50,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "subgoal_cmd_name": "subgoals",
        },
    )

    # task complete
    final_goal_reach = RewardTerm(
        func=mdp.task_goal_reach,
        weight=1,
        params={
            "pos_threshold": 0.39,
            "final_goal_reach_bonus": 200,
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
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
            "pos_threshold": 0.39,
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
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
class CabinetOpenEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for pick and place env."""

    # Scene settings
    scene: CabinetOpenSceneCfg = CabinetOpenSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.decimation = 2
        self.episode_length_s = 8.0
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation

        # Listens to the EE transforms
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

        # Listens to the handle transforms
        FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
        FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
        self.scene.handle_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
                prim_path="/Visuals/CabinetFrameTransformer"
            ),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                    name="drawer_handle_top",
                    offset=OffsetCfg(
                        pos=(0.305, 0.0, 0.01),
                        rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                    ),
                ),
            ],
        )
