from __future__ import annotations

import torch

from collections import defaultdict
from typing import Optional, Union, Mapping, Sequence, Any
from my_projects.utils.math import combine_frame_transforms, rotation_distance


class SubgoalPlanner:
    def __init__(
        self,
        subgoals: Mapping[str, Union[Sequence[Sequence[float]], torch.Tensor]],
        thresholds: Mapping[str, Union[Sequence[Sequence[float]], torch.Tensor]],
        num_envs: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

        assert self._get_dict_lengths(subgoals) == self._get_dict_lengths(
            thresholds
        ), "Input subgoals and subgoal_thresholds must have equal length for each item."

        self.single_subgoals = dict()
        for key, val in subgoals.items():
            self.single_subgoals[key] = (
                torch.tensor(val, device=self.device, dtype=torch.float32)
                if type(val) is not torch.Tensor
                else val.to(self.device).to(torch.float32)
            )

        self.subgoal_thresholds = dict()
        for key, val in thresholds.items():
            self.subgoal_thresholds[key] = (
                torch.tensor(val, device=self.device, dtype=torch.float32)
                if type(val) is not torch.Tensor
                else val.to(self.device).to(torch.float32)
            )

        self.single_subgoals_length = self._get_dict_lengths(self.single_subgoals)

        self.subgoals_and_thresholds = torch.zeros(
            (num_envs, sum(self.single_subgoals_length), 9), device=self.device, dtype=torch.float32
        )
        self.curr_subgoals_and_thresholds = torch.zeros((num_envs, 9), device=self.device, dtype=torch.float32)
        self.subgoals_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dones = torch.zeros_like(self.subgoals_indices, dtype=torch.bool)

    def _get_dict_lengths(self, input_dict: Mapping[str, Union[Sequence[Any], torch.Tensor]]) -> list[int]:
        length = []
        for _, value in input_dict.items():
            length.append(len(value))
        return length

    @property
    def shape(self) -> torch.Size:
        return self.subgoals_and_thresholds.shape

    @property
    def current_subgoals(self) -> torch.Tensor:
        return self.curr_subgoals_and_thresholds[:, :7]

    @property
    def current_thresholds(self) -> torch.Tensor:
        return self.curr_subgoals_and_thresholds[:, 7:9]

    def __len__(self) -> int:
        return self.num_envs

    def __getitem__(self, indices: Union[int, Sequence[int], slice]) -> torch.Tensor:
        return self.subgoals_and_thresholds[indices]

    def reset(self, objects_pose: Mapping[str, torch.Tensor], env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)

        # compute subgoal pose by current object pose
        subgoals = compute_subgoals(objects_pose, self.single_subgoals)

        # thresholds
        thresholds = (
            torch.cat([val for val in self.subgoal_thresholds.values()], dim=0).unsqueeze_(0).repeat(len(env_ids), 1, 1)
        )
        self.subgoals_and_thresholds[env_ids] = torch.cat((subgoals, thresholds), dim=-1)

        self.subgoals_indices[env_ids] = 0
        sample_indices = torch.tensor(
            [
                self.subgoals_indices[i] + i * sum(self.single_subgoals_length)
                for i in range(len(self.subgoals_indices))
            ],
            device=self.device,
            dtype=torch.long,
        )

        self.dones[env_ids] = 0

        self.curr_subgoals_and_thresholds = self.subgoals_and_thresholds.view(-1, 9)[sample_indices, :]

    def step(self, curr_ee_pose: torch.Tensor, include_quat: bool = False) -> None:
        "compute current subgoals by obs"
        # ee pos & rot
        curr_ee_pos = curr_ee_pose[:, :3]
        curr_ee_rot = curr_ee_pose[:, 3:7]

        # subgoal pos & rot
        curr_subgoal_pos = self.curr_subgoals_and_thresholds[:, :3]
        curr_subgoal_rot = self.curr_subgoals_and_thresholds[:, 3:7]

        # thresholds
        curr_pos_threshold = self.curr_subgoals_and_thresholds[:, 7]
        curr_rot_threshold = self.curr_subgoals_and_thresholds[:, 8]

        # whether to reach subgoals
        pos_dist = torch.norm(curr_ee_pos - curr_subgoal_pos, p=2, dim=-1)
        if include_quat:
            quat_dist = rotation_distance(curr_ee_rot, curr_subgoal_rot)
            subgoal_reached = (pos_dist <= curr_pos_threshold) & (quat_dist <= curr_rot_threshold)
        else:
            subgoal_reached = pos_dist <= curr_pos_threshold

        self.subgoals_indices = self.subgoals_indices + subgoal_reached
        self.dones = torch.where(
            self.subgoals_indices == sum(self.single_subgoals_length),
            torch.ones_like(self.dones),
            torch.zeros_like(self.dones),
        )

        self.subgoals_indices.clamp_(max=sum(self.single_subgoals_length) - 1)

        sample_indices = torch.tensor(
            [
                self.subgoals_indices[i] + i * sum(self.single_subgoals_length)
                for i in range(len(self.subgoals_indices))
            ],
            device=self.device,
            dtype=torch.int32,
        )
        self.curr_subgoals_and_thresholds = self.subgoals_and_thresholds.view(-1, 9)[sample_indices, :]


def compute_subgoals(
    objects_pose: Mapping[str, torch.Tensor], single_subgoals: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    # subgoals
    subgoals_dict = defaultdict(list)
    for key, val in single_subgoals.items():
        object_pose = objects_pose[key]
        for i in range(len(val)):
            subgoals_dict[key].append(
                torch.cat(
                    combine_frame_transforms(
                        t01=object_pose[:, :3],
                        q01=object_pose[:, 3:7],
                        t12=val[i, :3].repeat(object_pose.shape[0], 1),
                        q12=val[i, 3:7].repeat(object_pose.shape[0], 1),
                    ),
                    dim=-1,
                )
            )
    subgoals_list = []
    for val in subgoals_dict.values():
        subgoals_list += val

    subgoals = torch.stack(subgoals_list, dim=1)

    return subgoals
