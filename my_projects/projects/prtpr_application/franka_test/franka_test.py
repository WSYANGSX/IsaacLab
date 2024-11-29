from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Prtpr agent test on Franka Panda.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import pandas as pd
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    Articulation,
)
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from my_projects.projects.prtpr_application.prtpr_agent import PrtprAgent
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.sensors import FrameTransformerCfg, FrameTransformer
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform, quat_from_angle_axis, quat_mul
from omni.isaac.core.utils.torch import torch_rand_float
from my_projects.utils.math import rotation_distance

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip


marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """config for a franka scene"""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # light
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
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
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg.replace(prim_path="/Visuals/EEFrameTransformer"),
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


def generate_random_quat_with_z_in_hyposphere(num_samples: int, device: str):
    rot_x = quat_from_angle_axis(
        torch.rand(num_samples, device=device) * torch.pi + torch.pi / 2,
        torch.tensor((1.0, 0.0, 0.0), device=device).repeat(num_samples, 1),
    )
    rot_y = quat_from_angle_axis(
        (torch.rand(num_samples, device=device) * 2 - 1) * torch.pi / 2,
        torch.tensor((0.0, 1.0, 0.0), device=device).repeat(num_samples, 1),
    )
    rot_z = quat_from_angle_axis(
        torch.rand(num_samples, device=device),
        torch.tensor((0.0, 0.0, 1.0), device=device).repeat(num_samples, 1),
    )

    return quat_mul(rot_z, quat_mul(rot_y, rot_x))


def reset_target_pose(
    scene: InteractiveScene, sim: sim_utils.SimulationContext, target: VisualizationMarkers
) -> tuple[torch.Tensor, torch.Tensor]:
    # reset target position
    new_pos = torch_rand_float(-1.0, 1.0, (scene.num_envs, 3), device=sim.device)
    new_pos[:, 0] = torch.abs(new_pos[:, 0]) * 0.3 + 0.35
    new_pos[:, 1] = new_pos[:, 1] * 0.3
    new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.35) + 0.15

    # reset target rotation
    new_quat = generate_random_quat_with_z_in_hyposphere(scene.num_envs, device=sim.device)

    # update target pose
    target.visualize(new_pos + scene.env_origins, new_quat)
    return new_pos, new_quat


def reset_robot_states(
    robot: Articulation,
    scene: InteractiveScene,
    sim: sim_utils.SimulationContext,
    robot_dof_lower_limits: torch.Tensor,
    robot_dof_upper_limits: torch.Tensor,
) -> None:
    joint_pos = robot.data.default_joint_pos + sample_uniform(
        -0.125,
        0.125,
        (scene.num_envs, robot.num_joints),  # type: ignore
        sim.device,
    )
    joint_pos = torch.clamp(joint_pos, robot_dof_lower_limits, robot_dof_upper_limits)
    joint_vel = torch.zeros_like(joint_pos)
    robot.set_joint_position_target(joint_pos)  # type: ignore
    robot.write_joint_state_to_sim(joint_pos, joint_vel)  # type: ignore


def get_observations(
    robot: Articulation,
    ee_frame: FrameTransformer,
    hand_link_idx: int,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist: list,
    rot_dist: list,
):
    robot_dof_pos = robot.data.joint_pos[:, :7]
    robot_dof_vel = robot.data.joint_vel[:, :7]

    ee_pos, ee_rot = (
        ee_frame.data.target_pos_source[:, 0, :],
        ee_frame.data.target_quat_source[:, 0, :],
    )

    ee_lin_vel = robot.data.body_lin_vel_w[:, hand_link_idx, :]

    ee_target_dist = torch.norm(target_pos - ee_pos, p=2, dim=-1, keepdim=True)
    ee_target_rot_dist = rotation_distance(target_rot, ee_rot).view(-1, 1)

    dist.append(
        ee_target_dist.cpu().view(
            -1,
        )
    )
    rot_dist.append(
        ee_target_rot_dist.cpu().view(
            -1,
        )
    )

    return torch.cat(
        (
            ee_pos,
            ee_rot,
            target_pos,
            target_rot,
            ee_target_dist,
            ee_target_rot_dist,
            robot_dof_pos,
            robot_dof_vel,
            ee_lin_vel,
        ),
        dim=-1,
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
):
    """run the simulation loop"""

    # scene objects
    robot: Articulation = scene["robot"]
    ee_frame: FrameTransformer = scene["ee_frame"]

    # target
    target_cfg = VisualizationMarkersCfg(
        prim_path="/Visual/Target",
        markers={
            "target": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )
        },
    )
    target: VisualizationMarkers = VisualizationMarkers(target_cfg)  # type: ignore

    # similation parameters
    sim_dt = sim.get_physics_dt()
    decimation = 4
    render_interval = decimation
    count = 0
    sim_step_counter = 0
    file_num = 0

    # specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    arm_joint_idx = robot_entity_cfg.joint_ids
    hand_link_idx = robot_entity_cfg.body_ids[0]  # type: ignore

    # load policy
    checkpoint_path = "/home/yangxf/my_projects/IsaacLab/logs/rl_games/franka_prtpr_jointspace_direct/v2/nn/last_franka_prtpr_jointspace_direct_ep_2700_rew_4048.7012"
    prtpr_agent = PrtprAgent("Isaac-Franka_Prtpr-Direct-JointSpace-v0", sim.device, checkpoint_path)

    # robot parameters
    action_scale = torch.tensor(prtpr_agent.env_cfg.action_scale, device=sim.device)  # type: ignore
    robot_dof_lower_limits = robot.data.soft_joint_pos_limits[0, :, 0].to(device=sim.device)
    robot_dof_upper_limits = robot.data.soft_joint_pos_limits[0, :, 1].to(device=sim.device)
    robot_dof_speed_scales = torch.ones_like(robot_dof_lower_limits[: len(arm_joint_idx)])  # type: ignore

    # data buffer
    dist = []
    rot_dist = []

    # simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 499 == 0:
                if len(dist) != 0 and len(rot_dist) != 0:
                    with pd.ExcelWriter(
                        "my_projects/projects/prtpr_application/data/dist.xlsx",
                        engine="openpyxl",
                        mode="a",
                        if_sheet_exists="replace",
                    ) as writer:
                        ee_target_dist_df = pd.DataFrame(np.array(dist))
                        ee_target_rot_dist_df = pd.DataFrame(np.array(rot_dist))

                        ee_target_dist_df.to_excel(writer, sheet_name=f"ee_target_dist_{file_num}", index=True)
                        ee_target_rot_dist_df.to_excel(writer, sheet_name=f"ee_target_rot_dist{file_num}", index=True)

                count = 0
                file_num += 1
                dist.clear()
                rot_dist.clear()

                # reset robot joint states
                reset_robot_states(robot, scene, sim, robot_dof_lower_limits, robot_dof_upper_limits)

                # reset target pose
                target_pos, target_rot = reset_target_pose(scene, sim, target)

                # clear internal buffers
                scene.reset()  # reset对scene中的rigidobject、articulation、sensors分别进行reset,重置其actuator等内部项
                print("[INFO]: resetting robot state...")

                # reset()之后如果想要获取数据，需要进行step()
                sim.step()
                scene.update(sim_dt)

            # get observations
            obs = get_observations(robot, ee_frame, hand_link_idx, target_pos, target_rot, dist, rot_dist)

            if prtpr_agent.has_batch_dimension is False:
                prtpr_agent.get_batch_size(obs)
            actions = prtpr_agent.get_action(obs, is_deterministic=True)

            arm_targets = torch.clamp(
                robot.data.joint_pos[:, :7] + actions * sim_dt * decimation * robot_dof_speed_scales * action_scale,
                robot_dof_lower_limits[:7],
                robot_dof_upper_limits[:7],
            )

            # apply actions
            for _ in range(decimation):
                sim_step_counter += 1
                robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_idx)
                scene.write_data_to_sim()
                sim.step(render=False)
                if sim_step_counter % render_interval == 0:
                    sim.render()
                # update buffers at sim dt
                scene.update(dt=sim_dt)
            count += 1


def main():
    sim_cfg = SimulationCfg(device="cuda:0", dt=1 / 120)
    sim: SimulationContext = sim_utils.SimulationContext(sim_cfg)  # type: ignore

    sim.set_camera_view(eye=(2.5, 0.0, 4.0), target=(0.0, 0.0, 2.0))

    scene_cfg = FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
