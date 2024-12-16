import argparse
from omni.isaac.lab.app import AppLauncher

arg_parser = argparse.ArgumentParser(description="5.interacting with an articulation")
# user define cli
...
# app relative cli
...
AppLauncher.add_app_launcher_args(arg_parser)
arg_cli = arg_parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(arg_cli)
simulation_app = app_launcher.app


import torch

import omni.isaac.core.utils.prims as prims_untils
import omni.isaac.lab.sim as sim_untils
from omni.isaac.lab.assets.articulation import Articulation, ArticulationCfg
from omni.isaac.lab.sim import SimulationContext, SimulationCfg

# pre-defined configs
from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG


# 任务逻辑
# 1.design scene
def design_scene() -> tuple[dict[str, Articulation], list[list[float]]]:
    # ground plane
    ground_cfg = sim_untils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # light
    light_cfg = sim_untils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # create separate groups called "origin1", "origin2", "origin3"
    origins = [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        prims_untils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
    # add articulation in each origin
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Cartpole"
    cartpole = Articulation(cartpole_cfg)

    scence_entities = {"cartpole": cartpole}
    return scence_entities, origins


# 2.run simulation step
def run_simulator(
    sim: SimulationContext,
    scene_entities: dict[str, Articulation],
    origins: torch.Tensor,
):
    # 仿真参数
    sim_dt = sim.get_physics_dt()
    count = 0
    sim_time = 0.0

    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            count = 0
            cartpole_root_state = scene_entities["cartpole"].data.default_root_state
            cartpole_root_state[:, :3] += origins
            scene_entities["cartpole"].write_root_state_to_sim(cartpole_root_state)
            # set joint with some noise
            joint_pos, joint_vel = (
                scene_entities["cartpole"].data.default_joint_pos.clone(),
                scene_entities["cartpole"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene_entities["cartpole"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene_entities["cartpole"].reset()
            print("[INFO]: resetting robot state...")

        # apply random actions
        efforts = torch.randn_like(scene_entities["cartpole"].data.joint_pos) * 5.0
        # 应用到cartpole上
        scene_entities["cartpole"].set_joint_effort_target(efforts)
        # 将力信息写入sim
        scene_entities["cartpole"].write_data_to_sim()

        sim.step()
        count += 1
        sim_time += sim_dt

        scene_entities["cartpole"].update(sim_dt)


def main():
    sim_cfg = SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)

    # set camera
    sim.set_camera_view(eye=[2.5, 0.0, 4.0], target=[0.0, 0.0, 2.0])

    # design scene
    scene_entities, origins = design_scene()
    origins = torch.tensor(origins, device=sim.device)

    # reset sim
    sim.reset()
    print("[INFO]: setup complete...")

    # run simulation
    run_simulator(sim, scene_entities, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
