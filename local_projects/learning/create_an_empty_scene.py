import argparse

from omni.isaac.lab.app import AppLauncher

"""首先登录app"""

# create argparser
parser = argparse.ArgumentParser(description="Tutorial 1")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""任务逻辑实现部分"""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext


# 主函数
def main():
    # 初始化仿真参数
    sim_cfg = SimulationCfg(dt=0.01, use_fabric=True, substeps=2)
    sim = SimulationContext(sim_cfg)

    # 设置摄像机
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 运行仿真
    sim.reset()
    print("[INFO]: Setup complete...")
    
    # 时间步管理
    while simulation_app.is_running():
        # 执行物理步
        sim.step()
        

if __name__=="__main__":
    # 执行主函数
    main()
    # 关闭sim app
    simulation_app.close()