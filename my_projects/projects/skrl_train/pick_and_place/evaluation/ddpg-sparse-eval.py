import os 

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.Tanh(),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(
            torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        ), {}


# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Franka-Cabinet-Succ-Direct-v0", num_envs=1024)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models1 = {}
models1["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models1["target_policy"] = DeterministicActor(
    env.observation_space, env.action_space, device
)
models1["critic"] = Critic(env.observation_space, env.action_space, device)
models1["target_critic"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(
    theta=0.15, sigma=0.1, base_scale=0.5, device=device
)
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0  
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/Isaac-Franka-Cabinet-Succ-Direct-DDPG-Sparse"

agent = DDPG(
    models=models1,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


models_path = "./runs/torch/Isaac-Franka-Cabinet-Succ-Direct-DDPG-Sparse/5/checkpoints"
models_list = os.listdir(models_path)
sorted_model_names = sorted(models_list, key=lambda x: int(x.split("_")[1].split(".")[0]))

succ_rate = []

for model in sorted_model_names:
    agent.load(os.path.join(models_path, model))

    states, infos = env.reset()

    for i in range(500):  # env eposide-length setting
        # state-preprocessor + policy
        with torch.no_grad():
            states = agent._state_preprocessor(states)
            actions = agent.policy.act({"states": states}, role="policy")[0]

        # step the environment
        next_states, rewards, terminated, truncated, infos = env.step(actions)

        # render the environment
        env.render()

        # check for termination/truncation
        if terminated.any() or truncated.any():
            states, infos = env.reset()
        else:
            states = next_states

    success = env.success
    succ_rate.append((sum(success) / env.num_envs).item())

print(succ_rate)
env.close()
