import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser(description="2.spawning prims into the scene")
AppLauncher.add_app_launcher_args(arg_parser)
arg_cli = arg_parser.parse_args()
app_launcher = AppLauncher(arg_cli)
simulation_app = app_launcher.app

# 任务实现区域
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


# 设计任务场景
def design_scene():
    """designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGorundPlane", ground_cfg)

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform", translation=[10.0, 10.0, 10.0])
    # 产生一个红色cone
    cone_cfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cone_cfg.func("/World/Objects/Cone1", cone_cfg, translation=(-1.0, 1.0, 1.0))
    cone_cfg.func("/World/Objects/Cone2", cone_cfg, translation=(1.0, -1.0, 1.0))

    # 产生一个绿色cone，包含colliders和rigid body
    cone_rigid_cfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cone_rigid_cfg.func(
        "/World/Objects/RigidCone",
        cone_rigid_cfg,
        translation=(0.0, 0.0, 2.0),
        orientation=(0.5, 0.0, 0.5, 0.0),
    )

    # 产生一个桌子from usd file
    table_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    )
    table_cfg.func("/World/Objects/Table", table_cfg, translation=(0.0, 0.0, 1.05))

    # 产生灯光
    distant_light_cfg = sim_utils.DistantLightCfg(
        intensity=3000.0, color=(0.75, 0.75, 0.75)
    )
    distant_light_cfg.func(
        "/World/DistantLight", distant_light_cfg, translation=(1.0, 0.0, 10.0)
    )


# 主函数
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, substeps=2)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 设置摄像头
    sim.set_camera_view((2.0, 0.0, 2.5), [-0.5, 0.0, 0.5])

    # design scene by adding assets to it
    design_scene()

    # play simulation
    sim.reset()
    print("[INFO]: setup complete...")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
