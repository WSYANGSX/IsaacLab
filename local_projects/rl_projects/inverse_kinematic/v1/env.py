import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser("franka forward kinematic scene.")
# user cli
arg_parser.add_argument("--num_envs", type=int, default=2, help="num of envs")

AppLauncher.add_app_launcher_args(arg_parser)
arg_cli = arg_parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(arg_cli)
simulation_app = app_launcher.app


# 任务逻辑
import torch
import omni.isaac.lab.sim as sim_untils
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import unscale_transform
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
from omni.isaac.lab.utils.math import euler_xyz_from_quat

# predefined configs
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG


FRANKA_PANDA_HIGH_PD_CFG.spawn.articulation_props.solver_position_iteration_count = 100
FRANKA_PANDA_HIGH_PD_CFG.spawn.articulation_props.solver_velocity_iteration_count = 100


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """config for a cartpole scene"""

    # ground plane
    ground_cfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_untils.GroundPlaneCfg()
    )

    # light
    light_cfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_untils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # articulation
    franka: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.copy().replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """run the simulation loop"""
    robot: Articulation = scene["franka"]
    robot_cfg = SceneEntityCfg("franka", body_names=["panda_hand"])
    robot_cfg.resolve(scene)

    joint_limits = robot.root_physx_view.get_dof_limits()[0]
    print("joint_limits:", joint_limits)
    joint_lower_limits, joint_upper_limits = joint_limits.to(sim.device).T

    # 仿真参数
    sim_dt = sim.get_physics_dt()
    count = 0

    with open(
        "/media/yangxf/杨晓帆//data.txt",
        "a",
    ) as f:
        # simulation loop
        while simulation_app.is_running():
            # reset
            if count == 0:
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.clone(),
                    robot.data.default_joint_vel.clone(),
                )

                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                # clear internal buffers
                scene.reset()
                print("[INFO]: resetting robot state...")

            rand_float = (
                torch.rand(
                    (arg_cli.num_envs, 9), device=sim.device, dtype=torch.float32
                )
                * 2
                - 1
            )
            joint_pos = unscale_transform(
                rand_float, joint_lower_limits, joint_upper_limits
            )
            joint_pos[:, 7:9] = 0.0

            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            sim.step()
            count += 1
            scene.update(sim_dt)

            ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0]] - scene.env_origins
            ee_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
            euler_angles = euler_xyz_from_quat(ee_quat)
            _x_angles = euler_angles[0].view(-1, 1)
            _y_angles = euler_angles[1].view(-1, 1)
            _z_angles = euler_angles[2].view(-1, 1)
            euler_angles = torch.cat((_x_angles, _y_angles, _z_angles), dim=-1)

            data = torch.cat((joint_pos, ee_pos, ee_quat, euler_angles), dim=-1)
            print(data)

            # 记录数据
            for row in data:
                row_str = " ".join(map(str, row.tolist()))
                f.write(row_str + "\n")


def main():
    sim_cfg = SimulationCfg(device="cuda:0", use_gpu_pipeline=True, dt=1 / 60)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = FrankaSceneCfg(num_envs=arg_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
