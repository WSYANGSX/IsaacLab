from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="prtpr model test on ur10.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--activate_virtual_cube",
    type=bool,
    default=False,
    help="Number of environments to spawn.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import math
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.assets import (
    RigidObjectCfg,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    Articulation,
)
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import (
    quat_mul,
    quat_from_angle_axis,
    normalize,
    quat_conjugate,
    random_orientation,
    unscale_transform,
)
from omni.isaac.lab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from my_project.rl_projects.prtpr_application.prtpr_model import PrtprModel

# predefined configs
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """config for a franka scene"""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    franka: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.copy().replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # target cube
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                kinematic_enabled=True,  # 设置成kinematic_enabled=True意味着无法进行位置更改
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # visual cube
    inductor: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/visual_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
):
    """run the simulation loop"""

    # scene objects
    robot: Articulation = scene["franka"]
    target: RigidObject = scene["target"]
    if hasattr(scene.cfg, "inductor"):
        inductor: RigidObject = scene["inductor"]

    # scene parameters
    action_limits = torch.tensor(
        [
            [-math.pi, math.pi],
            [-math.pi / 2, math.pi / 2],
            [-0.02, 0.02],
            [-math.pi / 90, math.pi / 90],
        ],
        dtype=torch.float32,
        device=sim.device,
    )
    action_lower_limits, action_upper_limits = torch.t(action_limits)

    # similation parameters
    sim_dt = sim.get_physics_dt()
    count = 0

    # specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg(
        "franka", joint_names=["panda_joint.*"], body_names=["panda_hand"]
    )
    robot_entity_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # create controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

    # load policy
    policy_path = "/home/yangxf/Ominverse_RL_platform/IsaacLab/logs/rl_games/prtpr/3/"
    policy = PrtprModel(policy_path=policy_path)
    policy.reset()
    if policy.is_rnn:
        policy.init_rnn()
    batch_size = 1

    # simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0

                # reset robot joint states
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.clone(),
                    robot.data.default_joint_vel.clone(),
                )
                joint_pos += torch.rand_like(joint_pos) * 0.1
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.set_joint_position_target(joint_pos)

                # reset target pose
                new_pos = random_position(num=args_cli.num_envs, device=sim.device)
                new_pos += scene.env_origins
                new_rot = random_orientation(num=args_cli.num_envs, device=sim.device)
                new_pose = torch.cat((new_pos, new_rot), dim=-1)
                target.write_root_pose_to_sim(new_pose)

                # reset controller
                diff_ik_controller.reset()

                # clear internal buffers
                scene.reset()  # reset对scene中的rigidobject、articulation、sensors分别进行reset,重置其actuator等内部项
                print("[INFO]: resetting robot state...")

                # reset()之后如果想要获取数据，需要进行step()
                sim.step()
                scene.update(sim_dt)

                if hasattr(scene.cfg, "inductor"):
                    ee_pos = robot.data.body_state_w[
                        :, robot_entity_cfg.body_ids[0], 0:3
                    ]
                    ee_rot = robot.data.body_state_w[
                        :, robot_entity_cfg.body_ids[0], 3:7
                    ]
                    inductor.write_root_pose_to_sim(torch.cat((ee_pos, ee_rot), dim=-1))

            # get observations
            ee_pos = (
                robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3]
                - scene.env_origins
            )
            ee_rot = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 3:7]

            target_pos = target.data.root_pos_w - scene.env_origins
            target_rot = target.data.root_quat_w

            dist = torch.norm(
                target_pos - ee_pos,
                p=2,
                dim=-1,
                keepdim=True,
            )
            rot_diff = normalize(quat_mul(target_rot, quat_conjugate(ee_rot)))

            obs = torch.cat(
                (ee_pos, ee_rot, target_pos, target_rot, dist, rot_diff),  # type: ignore
                dim=-1,
            )

            if hasattr(scene.cfg, "inductor"):
                vc_pos = inductor.data.root_pos_w - scene.env_origins
                vc_rot = inductor.data.root_quat_w

                dist = torch.norm(
                    target_pos - vc_pos,
                    p=2,
                    dim=-1,
                    keepdim=True,
                )
                rot_diff = normalize(quat_mul(target_rot, quat_conjugate(vc_rot)))

                obs = torch.cat(
                    (vc_pos, vc_rot, target_pos, target_rot, dist, rot_diff),  # type: ignore
                    dim=-1,
                )

            if not policy.has_batch_dimension:
                print("*********** init batch size ***********")
                policy.get_batch_size(obses=obs, batch_size=batch_size)

            # compute rot axis
            w = rot_diff[:, 0]
            a = rot_diff[:, 1]
            b = rot_diff[:, 2]
            c = rot_diff[:, 3]
            rot_axis = torch.cat(
                [
                    torch.reshape(a / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
                    torch.reshape(b / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
                    torch.reshape(c / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
                ],
                dim=-1,
            )

            # compute actions
            raw_act = policy.get_action(obs, is_deterministic=True)
            act = raw_act.clone().clamp(-1.0, 1.0)
            act = unscale_transform(act, action_lower_limits, action_upper_limits)

            # apply actions
            if hasattr(scene.cfg, "inductor"):
                # change virtual_cube_pose
                next_pose = compute_next_pose(vc_pos, vc_rot, rot_axis, act)
                next_pose[:, :3] = 0.7 * vc_pos + 0.3 * next_pose[:, :3]
                next_pose[:, :3] += scene.env_origins
                inductor.write_root_pose_to_sim(next_pose)
            else:
                # compute next ee pose
                next_pose = compute_next_pose(ee_pos, ee_rot, rot_axis, act)

            # inverse kinematic
            # obtain quantities from simulation
            target_local_pose = next_pose
            target_local_pose[:, :3] -= scene.env_origins

            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
            ]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ik_command = torch.where(
                torch.norm(
                    target_pos - ee_pos,
                    p=2,
                    dim=-1,
                    keepdim=True,
                )
                >= 0.05,
                target_local_pose,
                torch.cat((target_pos, target_rot), dim=-1),
            )
            diff_ik_controller.set_command(ik_command)
            joint_pos_des = diff_ik_controller.compute(
                ee_pos, ee_rot, jacobian, joint_pos
            )
            robot.set_joint_position_target(
                joint_pos_des, joint_ids=robot_entity_cfg.joint_ids
            )
            scene.write_data_to_sim()

            sim.step()
            count += 1
            scene.update(sim_dt)


def main():
    sim_cfg = SimulationCfg(device="cuda:0", use_gpu_pipeline=True, dt=1 / 60)
    sim: SimulationContext = sim_utils.SimulationContext(sim_cfg)  # type: ignore

    sim.set_camera_view(eye=(2.5, 0.0, 4.0), target=(0.0, 0.0, 2.0))

    scene_cfg = FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    if not args_cli.activate_virtual_cube:
        del scene_cfg.inductor
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: setup complete...")

    run_simulator(sim, scene)


"""
Helper function
"""


def compute_next_pose(
    current_pos: torch.Tensor,
    current_rot: torch.Tensor,
    rot_axis: torch.Tensor,
    act: torch.Tensor,
) -> torch.Tensor:
    thetas1 = act[:, 0]
    thetas2 = act[:, 1]
    dist = torch.abs(act[:, 2])

    _x = dist * torch.cos(thetas2) * torch.cos(thetas1)
    _y = dist * torch.cos(thetas2) * torch.sin(thetas1)
    _z = dist * torch.sin(thetas2)

    next_pos = current_pos + torch.stack((_x, _y, _z), dim=-1)

    angle = act[:, 3]
    rot = quat_from_angle_axis(angle, rot_axis)
    next_rot = quat_mul(rot, current_rot)
    next_pose = torch.cat((next_pos, next_rot), dim=-1)

    return next_pose


@torch.jit.script
def random_position(num: int, device: str) -> torch.Tensor:
    rand_float = torch.randn((num, 3), device=device, dtype=torch.float)

    new_pos = torch.zeros_like(rand_float, device=device)
    new_pos[:, 0] = rand_float[:, 0] * 0.1 + 0.2 * torch.sign(rand_float[:, 0])
    new_pos[:, 1] = rand_float[:, 1] * 0.1 + 0.2 * torch.sign(rand_float[:, 1])
    new_pos[:, 2] = torch.abs(rand_float[:, 2] * 0.3) + 0.1

    return new_pos


if __name__ == "__main__":
    main()
    simulation_app.close()
