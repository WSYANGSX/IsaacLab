import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser("6.using the interactive scene")
# user cli
arg_parser.add_argument("--num_envs", type=int, default=3, help="num of envs")
# app cli
...
AppLauncher.add_app_launcher_args(arg_parser)
arg_cli = arg_parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(arg_cli)
simulation_app = app_launcher.app


# 任务逻辑
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.assets import RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """config for a cartpole scene"""

    # ground plane
    ground_cfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.675)),
    )

    # light
    light_cfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/table/table/parts/Part_1_JHD.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.96, 0.87, 0.702)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, -0.04)),
    )
    
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block_instanceable.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, -0.3, 2.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    
    # robot
    robot = FRANKA_PANDA_CFG.copy().replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """run the simulation loop"""
    table = scene["table"]
    # 仿真参数
    sim_dt = sim.get_physics_dt()
    count = 0

    # simulation loop
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            count = 0
            root_state = table.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            table.write_root_state_to_sim(root_state)

            # clear internal buffers
            scene.reset()
            print("[INFO]: resetting robot state...")

        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = CartpoleSceneCfg(num_envs=arg_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
