import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser(description="4.interacting with rigid bodies")
# user define cli
arg_parser.add_argument(
    "--cone_radius", type=float, default=0.5, help="the radius of rigid cones"
)
arg_parser.add_argument(
    "--cone_height", type=float, default=1, help="the height of rigid cones"
)
# app relative cli
arg_parser.add_argument(
    "--width", type=int, default=1980, help="the width of app viewport"
)
arg_parser.add_argument(
    "--height", type=int, default=1020, help="the height of app viewport"
)

# 向app_launcher添加参数关键字
AppLauncher.add_app_launcher_args(arg_parser)

# 解析参数并创建app
arg_cli = arg_parser.parse_args()
app_launcher = AppLauncher(arg_cli)
simulation_app = app_launcher.app


############################ 任务逻辑 ############################
import torch
from typing import Any
import omni.isaac.lab.sim as sim_untils
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.lab.assets.rigid_object import RigidObjectCfg, RigidObject
from omni.isaac.lab.utils import math as math_untils

# 1.设计任务场景
def design_scene() -> Any:
    # ground plane
    ground_cfg = sim_untils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # rigid bodies
    origins = [[2.5, 2.5, 0.5], [-2.5, 2.5, 0.5], [2.5, -2.5, 0.5], [-2.5, -2.5, 0.5]]
    for i, origins in enumerate(origins):
        create_prim(f"/World/Rigid{i}", "Xform", translation=origins)

    rigid_objects_cfg = RigidObjectCfg(
        prim_path="/World/Rigid.*/Cone",
        spawn=sim_untils.ConeCfg(
            radius=arg_cli.cone_radius, 
            height=arg_cli.cone_height,
            mass_props=sim_untils.MassPropertiesCfg(1.0),
            rigid_props=sim_untils.RigidBodyPropertiesCfg(),
            collision_props=sim_untils.CollisionPropertiesCfg(),
            visual_material=sim_untils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cones = RigidObject(rigid_objects_cfg)

    # distant light
    distant_light_cfg = sim_untils.DistantLightCfg(
        intensity=3000.0, color=(0.8, 0.8, 0.8)
    )
    distant_light_cfg.func("/World/DistantLight", distant_light_cfg)

    scene_entities = {"cones": cones}
    return scene_entities, origins


# simulatuion loop
def run_simulation(
    sim: sim_untils.SimulationContext,
    scene_entities: dict[str, RigidObject],
    origins: torch.Tensor,
):
    # 仿真参数
    sim_dt = sim.get_physics_dt()
    count = 0
    sim_times = 0

    while simulation_app.is_running():
        # reset cones
        if count % 250 == 0:
            count = 0
            sim_times = 0.0
            cones_root_state = scene_entities["cones"].data.default_root_state.clone()
            cones_root_state[:, :3] += origins
            cones_root_state[:, :3] += math_untils.sample_cylinder(
                radius=0.1,
                h_range=(0.25, 0.5),
                size=scene_entities["cones"].num_instances,
                device=scene_entities["cones"].device,
            )
            scene_entities["cones"].write_root_state_to_sim(cones_root_state)
            # reset buffers
            scene_entities["cones"].reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")

        scene_entities["cones"].write_data_to_sim()
        sim.step()
        count += 1
        sim_times += sim_dt
        scene_entities["cones"].update(sim_dt)

        if count % 50 == 0:
            print(
                f"Root position (in world): {scene_entities['cones'].data.root_state_w[:, :3]}"
            )


# main函数
def main():
    sim_cfg = sim_untils.SimulationCfg(dt=0.01, substeps=2)
    sim = sim_untils.SimulationContext(sim_cfg)

    # set camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0, 0, 0, 0, 0, 0])

    # design scene
    scene_entities, origins = design_scene()
    origins = torch.tensor(origins, device=sim.device)
    
    # reset sim
    sim.reset()
    print("[INFO]: setup complete...")

    # sim loop
    run_simulation(sim, scene_entities, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
