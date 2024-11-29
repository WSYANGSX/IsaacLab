import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser("use quat to make coordinate transformation ")
# user cli
arg_parser.add_argument("--num_envs", type=int, default=1, help="num of envs")
# app cli
...
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
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.markers import (
    FRAME_MARKER_CFG,
    VisualizationMarkersCfg,
    VisualizationMarkers,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import (
    random_orientation,
    quat_mul,
    quat_conjugate,
    normalize,
    axis_angle_from_quat,
)
from my_project.rl_projects.utils.math import rotation_distance
from omni.isaac.core.utils.torch import quat_from_angle_axis


@configclass
class FrameMakerSceneCfg(InteractiveSceneCfg):
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


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    init_frame: VisualizationMarkers,
    target_frame: VisualizationMarkers,
    middle_frame: VisualizationMarkers,
):
    """run the simulation loop"""
    init_frame = init_frame
    target_frame = target_frame
    middle_frame = middle_frame

    init_frame_pos = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    init_frame_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)

    target_frame_pos = torch.tensor([[10, 10, 10]], device=sim.device)
    target_frame_quat = random_orientation(1, device=sim.device)

    steps = 500

    pos_delta = (target_frame_pos - init_frame_pos) / steps

    # 仿真参数
    sim_dt = sim.get_physics_dt()
    count = 0

    # simulation loop
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            count = 0

            middle_frame_pos = init_frame_pos
            middle_frame_quat = init_frame_quat

            init_frame.visualize(
                translations=init_frame_pos, orientations=init_frame_quat
            )
            target_frame.visualize(
                translations=target_frame_pos, orientations=target_frame_quat
            )
            middle_frame.visualize(
                translations=init_frame_pos, orientations=init_frame_quat
            )

            # clear internal buffers
            scene.reset()
            print("[INFO]: resetting robot state...")

        print(middle_frame_quat)
        rot_diff = normalize(
            quat_mul(target_frame_quat, quat_conjugate(middle_frame_quat))
        )
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
        _angle = rotation_distance(middle_frame_quat, target_frame_quat)

        angle_delta = _angle / (steps - count)
        quat_delat = quat_from_angle_axis(angle_delta, axis=rot_axis)
        middle_frame_pos = middle_frame_pos + pos_delta
        middle_frame_quat = quat_mul(middle_frame_quat, quat_delat)

        middle_frame.visualize(
            translations=middle_frame_pos, orientations=middle_frame_quat
        )

        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = FrameMakerSceneCfg(num_envs=arg_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # init frame
    init_frame: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy().replace(
        prim_path="/Visual/Init_frame"
    )

    # target frame
    target_frame: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy().replace(
        prim_path="/Visual/Target_frame"
    )

    # middle frame
    middle_frame: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy().replace(
        prim_path="/Visual/Middle_frame"
    )

    init_frame = VisualizationMarkers(init_frame)
    target_frame = VisualizationMarkers(target_frame)
    middle_frame = VisualizationMarkers(middle_frame)

    sim.reset()
    print("[INFO]: setup complete...")

    run_simulator(sim, scene, init_frame, target_frame, middle_frame)


if __name__ == "__main__":
    main()
    simulation_app.close()
