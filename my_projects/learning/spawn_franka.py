import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser("6.using the interactive scene")
# user cli
arg_parser.add_argument("--num_envs", type=int, default=2, help="num of envs")
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
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.utils import configclass

# predefined configs
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
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
    cartpole: ArticulationCfg = FRANKA_PANDA_CFG.copy().replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """run the simulation loop"""
    robot: Articulation = scene["cartpole"]
    # 仿真参数
    sim_dt = sim.get_physics_dt()
    count = 0

    # simulation loop
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos)*0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            scene.reset()
            print("[INFO]: resetting robot state...")
            
        
        sim.step()
        count += 1
        scene.update(sim_dt)
        

def main():
    sim_cfg = SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)
    
    sim.set_camera_view([2.5,0.0,4.0],[0.0,0.0,2.0])
    
    scene_cfg = CartpoleSceneCfg(num_envs=arg_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[INFO]: setup complete...")
    
    run_simulator(sim, scene)
    

if __name__=="__main__":
    main()
    simulation_app.close()