from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Configuration for randomization."""

    pass


@configclass
class PrtprEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10
    num_actions = 4
    num_observations = 19
    num_states = 0
    asymmetric_obs = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        gravity=(0.0, 0.0, 0.0),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )

    # cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
        ),
    )

    # target
    target: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visual/markers",
        markers={
            "target": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
            )
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # rewards scales
    dist_reward_scale = -5
    rot_reward_scale = 0.1
    rot_eps = 0.1
    action_penalty_scale = 0
    eposide_lengths_penalty_scale = -0.001
    reach_target_bonus = 500
    dist_tolerance = 0.1
    rot_tolerance = 0.1

    # domain randomization config
    events: EventCfg = EventCfg()

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = (
        NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
        )
    )
