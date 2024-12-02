import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser(description="3.deep dive into applauncher")
# user define cli
arg_parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# simulationapp relative cli
arg_parser.add_argument("--width", type=int, default=1280, help="Width of the viewport and generated images.Defaults to 1280")
arg_parser.add_argument("--height", type=int, default=720, help="Height of the viewport and generated images.Defaults to 720")
AppLauncher.add_app_launcher_args(arg_parser)

arg_cli = arg_parser.parse_args()
app_launcher = AppLauncher(arg_cli)

simulation_app = app_launcher.app

# 任务逻辑
import omni.isaac.lab.sim as sim_utils

def design_scene():
    # ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    
    # cuboid
    cuboid_cfg = sim_utils.CuboidCfg(
        size=[arg_cli.size]*3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,1.0,1.0)),
    )
    cuboid_cfg.func("/World/Object", cuboid_cfg, translation=(0.0, 0.0, arg_cli.size/2))
    
    # distant light
    distant_light_cfg = sim_utils.DistantLightCfg(intensity=3000.0,color=(1.0,0.0,1.0))
    distant_light_cfg.func("/World/DistantLight", distant_light_cfg)
    
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, substeps=2)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 添加摄像机
    sim.set_camera_view([2.0,0.0,2.5],[-0.5,0.0,0.5])
    
    # design scene
    design_scene()
    
    # simulation play
    sim.reset()
    print("[INFO]: setup complete...")
    
    # physics control
    while simulation_app.is_running():
        sim.step()
        

if __name__=="__main__":
    main()
    simulation_app.close()    

