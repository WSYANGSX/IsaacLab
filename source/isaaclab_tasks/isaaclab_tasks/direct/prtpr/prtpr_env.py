from __future__ import annotations

import math
import torch

from typing import Sequence

from collections import deque
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaacsim.core.utils.torch import torch_rand_float
from isaaclab.utils.math import (
    quat_mul,
    quat_from_angle_axis,
    normalize,
    quat_conjugate,
)
from isaaclab.markers import VisualizationMarkers
from .prtpr_env_cfg import PrtprEnvCfg


class PrtprEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: PrtprEnvCfg

    def __init__(self, cfg: PrtprEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.action_limits = torch.tensor(
            [
                [-math.pi, math.pi],
                [-math.pi / 2, math.pi / 2],
                [-0.02, 0.02],
                [-math.pi / 90, math.pi / 90],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.action_lower_limits, self.action_upper_limits = torch.t(self.action_limits.to(self.device))

        # 单位向量
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.target = VisualizationMarkers(self.cfg.target)

        # buffer for next target
        self.current_target = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        # default target positions
        self.target_pos_l = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_rot_l = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.target_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_rot_w = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        # default cube positions
        self.cube_pos_l = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.cube_rot_l = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.cube_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.cube_rot_w = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        # rot axis
        self.rot_diff = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.rot_axis = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_three_successes_rate = deque(maxlen=3)
        # track goal resets
        # !!!!!!!!!!!!!!!!!!!!!! not bool type
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        print("*************** task initialed *****************")

    def _setup_scene(self):
        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # 计算下一坐标位置
        self.actions = actions.clone().clamp(-1.0, 1.0)

        actions = scale(self.actions, self.action_lower_limits, self.action_upper_limits)

        thetas1 = actions[:, 0]
        thetas2 = actions[:, 1]
        dist = torch.abs(actions[:, 2])

        _x = dist * torch.cos(thetas2) * torch.cos(thetas1)
        _y = dist * torch.cos(thetas2) * torch.sin(thetas1)
        _z = dist * torch.sin(thetas2)

        self.current_target[:, :3] = self._cube.data.root_pos_w + torch.stack((_x, _y, _z), dim=-1)

        angle = actions[:, 3]
        rot = quat_from_angle_axis(angle, self.rot_axis)
        self.current_target[:, 3:7] = quat_mul(rot, self.cube_rot_w)

    def _apply_action(self):
        # 应用坐标位置
        self._cube.write_root_pose_to_sim(root_pose=self.current_target)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # reset when time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # no terminated reset
        terminated = torch.zeros_like(time_out)

        return terminated, time_out

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._cube._ALL_INDICES

        # data for cube
        self.cube_pos_w[env_ids] = self._cube.data.root_pos_w[env_ids]
        self.cube_rot_w[env_ids] = self._cube.data.root_quat_w[env_ids]

        self.cube_pos_l[env_ids] = self.cube_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.cube_rot_l[env_ids] = self.cube_rot_w[env_ids]

        self.rot_diff[env_ids] = normalize(
            quat_mul(self.target_rot_w[env_ids], quat_conjugate(self.cube_rot_w[env_ids]))
        )
        w = self.rot_diff[env_ids][:, 0]
        a = self.rot_diff[env_ids][:, 1]
        b = self.rot_diff[env_ids][:, 2]
        c = self.rot_diff[env_ids][:, 3]
        self.rot_axis[env_ids] = torch.cat(
            [
                torch.reshape(a / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
                torch.reshape(b / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
                torch.reshape(c / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
            ],
            dim=-1,
        )

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        (total_reward, self.successes, self.reset_goal_buf) = compute_rewards(
            self.reset_goal_buf,
            self.episode_length_buf,
            self.successes,
            self.cube_pos_l,
            self.cube_rot_l,
            self.target_pos_l,
            self.target_rot_l,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.eposide_lengths_penalty_scale,
            self.cfg.reach_target_bonus,
            self.cfg.dist_tolerance,
            self.cfg.rot_tolerance,
        )

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        self._compute_intermediate_values(goal_env_ids)

        return total_reward

    def _reset_idx(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = self._cube._ALL_INDICES

        # resets rigid body attributes
        super()._reset_idx(env_ids)

        # reset object
        self._reset_cube_pose(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # curriculum
        self._curriculum()

        self.successes[env_ids] = 0
        self._compute_intermediate_values(env_ids)  # type: ignore

    def _curriculum(self):
        # 改变训练成功的难度
        succecc_num = torch.sum(self.successes)
        success_rate = succecc_num / self.num_envs
        self.last_three_successes_rate.append(success_rate)

        last_three_successes_rate = torch.tensor(list(self.last_three_successes_rate), device=self.device)
        print("last three successes rate: ", last_three_successes_rate)

        if all(last_three_successes_rate >= 2):
            print("******************** curriculum performed **************************")
            self.cfg.dist_tolerance *= 0.8

            if self.cfg.dist_tolerance < 0.01:
                self.cfg.dist_tolerance = 0.01

        print("current dist_tolerance: ", self.cfg.dist_tolerance)
        print("current rot_tolerance: ", self.cfg.rot_tolerance)

    def _reset_target_pose(self, env_ids):
        # reset target position
        new_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_pos[:, 0] = new_pos[:, 0] * 0.35 + 0.5 * torch.sign(new_pos[:, 0])
        new_pos[:, 1] = new_pos[:, 1] * 0.35 + 0.5 * torch.sign(new_pos[:, 1])
        new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.6) + 0.1

        # reset target rotation
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )

        # update target pose
        self.target_pos_l[env_ids] = new_pos
        self.target_rot_l[env_ids] = new_rot
        self.target_pos_w[env_ids] = new_pos + self.scene.env_origins[env_ids]
        self.target_rot_w[env_ids] = new_rot

        self.target.visualize(self.target_pos_w, self.target_rot_w)

        self.reset_goal_buf[env_ids] = 0

    def _reset_cube_pose(self, env_ids: Sequence[int]):
        # reset cube position
        new_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        new_pos[:, 0] = new_pos[:, 0] * 0.35 + 0.5 * torch.sign(new_pos[:, 0])
        new_pos[:, 1] = new_pos[:, 1] * 0.35 + 0.5 * torch.sign(new_pos[:, 1])
        new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.6) + 0.1

        # reset cube rotation
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )

        # update cube pose
        self.cube_pos_l[env_ids] = new_pos
        self.cube_rot_l[env_ids] = new_rot
        self.cube_pos_w[env_ids] = new_pos + self.scene.env_origins[env_ids]
        self.cube_rot_w[env_ids] = new_rot

        self._cube.write_root_pose_to_sim(
            torch.cat((self.cube_pos_w[env_ids], self.cube_rot_w[env_ids]), dim=-1),
            env_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = torch.cat(
            (
                self.cube_pos_l,
                self.cube_rot_l,
                self.target_pos_l,
                self.target_rot_l,
                torch.norm(
                    self.cube_pos_l - self.target_pos_l,
                    p=2,
                    dim=-1,
                    keepdim=True,
                ),
                normalize(quat_mul(self.target_rot_l, quat_conjugate(self.cube_rot_l))),
            ),
            dim=-1,
        )

        if self.cfg.asymmetric_obs:
            states = self._get_states()

        observations = {"policy": obs}

        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}

        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                self.cube_pos_l,
                self.cube_rot_l,
                self.target_pos_l,
                self.target_rot_l,
                torch.norm(
                    self.cube_pos_l - self.target_pos_l,
                    p=2,
                    dim=-1,
                    keepdim=True,
                ),
                normalize(quat_mul(self.target_rot_l, quat_conjugate(self.cube_rot_l))),
                self.actions,
            ),
            dim=-1,
        )
        return states


# 并非完全随机，因为先绕（0,1,0）进行旋转，然后绕（1,0,0）进行旋转，此时y轴上的点永远在zoy平面上，可以再加绕（0,0,1）的随机旋转表示完全旋转
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * math.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * math.pi, y_unit_tensor),
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def compute_rewards(
    reset_goal_buf: torch.Tensor,
    eposide_length_buf: torch.Tensor,
    successes: torch.Tensor,
    cube_pos: torch.Tensor,
    cube_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    eposide_length_penalty_scale: float,
    reach_target_bonus: float,
    dist_tolerance: float,
    rot_tolerance: float,
):
    target_dist = torch.norm(cube_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(cube_rot, target_rot)

    dist_rew = target_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # action 相关惩罚
    action_penalty = torch.sum(actions**2, dim=-1) * action_penalty_scale

    # eposide_length 相关惩罚
    eposide_length_penalty = eposide_length_buf * eposide_length_penalty_scale

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty + eposide_length_penalty

    # Find out which envs hit the target and update successes count
    dict_successes = torch.where(
        target_dist <= dist_tolerance,
        torch.ones_like(reset_goal_buf),
        torch.zeros_like(reset_goal_buf),
    )
    rot_successes = torch.where(
        torch.abs(rot_dist) <= rot_tolerance,
        torch.ones_like(reset_goal_buf),
        torch.zeros_like(reset_goal_buf),
    )
    target_resets = torch.where(
        dict_successes + rot_successes == 2,
        torch.ones_like(reset_goal_buf),
        torch.zeros_like(reset_goal_buf),
    )

    successes = successes + target_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(target_resets == 1, reward + reach_target_bonus, reward)

    return reward, successes, target_resets
