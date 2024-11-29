import sys
print(sys.path)

import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
arg_pareser = argparse.ArgumentParser(
    description="tutorial on running the cartpole RL env"
)
arg_pareser.add_argument("--num_envs", type=int, default=16, help="num of envs")

AppLauncher.add_app_launcher_args(arg_pareser)
args_cli = arg_pareser.parse_args()

app_launcher = AppLauncher(args_cli)
simulaion_app = app_launcher.app


# 任务逻辑
import torch

from my_project.learning.creating_a_managered_based_rl_evn import CartPoleEnvCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv


def main():
    env_cfg: CartPoleEnvCfg = CartPoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulation physics
    count = 0
    while simulaion_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Restting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            count += 1
            
    env.close()
    

if __name__=="__main__":
    main()
    simulaion_app.close()