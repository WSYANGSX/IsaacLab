import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser(description="7.manager based env")
arg_parser.add_argument(
    "--num_envs", type=int, default=15, help="num of envs to spawn."
)

# append Applauncher cli args
AppLauncher.add_app_launcher_args(arg_parser)
arg_cli = arg_parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(arg_cli)
simulation_app = app_launcher.app


#################### 任务逻辑实现 ####################
import math
import torch

from omni.isaac.lab.envs import mdp
from omni.isaac.lab.envs import ManagerBasedEnvCfg, ManagerBasedEnv
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import (
    CartpoleSceneCfg,
)


@configclass
class ActionCfg:
    """action specifications for the env"""

    joint_efforts = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=5.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for environment"""

    @configclass
    class PolicyCfg(ObsGroup):
        """obs for policy group"""

        # obs terms
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post__init(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """configuration for events"""

    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """configuration for the cartpole environment"""

    # scene setting
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # basic setting
    observations = ObservationsCfg()
    actions = ActionCfg()
    events = EventCfg()

    def __post_init__(self):
        """post  initialization"""
        # viewer setting
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]

        # step setting
        self.decimation = 4

        # simulation setting
        self.sim.dt = 0.005


def main():
    """main func"""
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = arg_cli.num_envs
    env = ManagerBasedEnv(env_cfg)

    # simulation physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the env
            obs, _ = env.step(joint_efforts)
            print(obs)
            count += 1
    
    env.close()
    

if __name__=="__main__":
    main()
    simulation_app.close()