import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewardTerm
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.multi_stage_manipulation.pick_and_place.mdp as mdp

###
# pre-defined franka pandas configs
###
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG


###
# scene define
###
@configclass
class PickAndPlaceCfg(InteractiveSceneCfg):
    """configuration for a franka pandas pick-and-place scene"""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # franka pandas
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )

    # cube
    cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.275, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )


###
# MDP settings
###


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.ArmActionCfg(
        policy_path="/home/yangxf/Ominverse_RL_platform/IsaacLab/logs/rl_games/prtpr/3/"
    )
    gripper_action: ActionTerm = mdp.GripperActionCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        cube_pos = ObsTerm(
            func=mdp.get_asset_local_pos,
            params={"asset_cfg": SceneEntityCfg("cube")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )

        cube_rot = ObsTerm(
            func=mdp.get_asset_local_rot,
            params={"asset_cfg": SceneEntityCfg("cube")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )

        ee_pose = ObsTerm(
            func=mdp.get_grip_point_local_pose,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", body_names=["panda_hand"])
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )

        goal_pose = ObsTerm(
            func=mdp.get_subgoal_local_pose,
        )

        # arm_joint_pos = ObsTerm(
        #     func=mdp.get_arm_position,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        # )

        # arm_joint_vel = ObsTerm(
        #     func=mdp.get_arm_velocity,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        # )

        gripper_joint_pos = ObsTerm(
            func=mdp.get_gripper_position,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        gripper_joint_vel = ObsTerm(
            func=mdp.get_gripper_velocity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
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
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
        },
    )

    # task complete
    final_goal_reach = RewardTerm(
        func=mdp.final_goal_reach,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "command_name": "ee_pose",
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # task complete
    task_complete = DoneTerm(
        mdp.task_complete, params={"asset_cfg": SceneEntityCfg("cube")}
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
        },
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class PickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for pick and place env."""

    # Scene settings
    scene: PickAndPlaceCfg = PickAndPlaceCfg(num_envs=4096, env_spacing=2.5)
    # Basic setting
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP setting
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 12.0
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
