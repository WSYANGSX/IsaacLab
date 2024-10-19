from __future__ import annotations

import torch

from collections import defaultdict
from typing import Optional, Union, Mapping, Sequence, Any
from local_projects.utils.math import combine_frame_transforms


class SubgoalPlanner:
    def __init__(
        self,
        subgoals: dict[str, Union[Sequence[Sequence[float]], torch.Tensor]],
        subgoal_thresholds: dict[str, Union[Sequence[float], torch.Tensor]],
        num_envs: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        assert self._get_dict_lengths(subgoals) == self._get_dict_lengths(
            subgoal_thresholds
        ), "Input subgoals and subgoal_thresholds must have equal length for each item."

        self.single_subgoals = dict()
        for key, val in subgoals.items():
            self.single_subgoals[key] = torch.tensor(val) if type(val) is not torch.Tensor else val

        self.subgoal_thresholds = dict()
        for key, val in subgoal_thresholds.items():
            self.subgoal_thresholds[key] = torch.tensor(val) if type(val) is not torch.Tensor else val

        self.single_subgoals_length = self._get_dict_lengths(self.single_subgoals)

        self.num_envs = num_envs
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

        self.subgoals = torch.zeros(
            (num_envs, sum(self.single_subgoals_length), 8), device=self.device, dtype=torch.float32
        )

    def _get_dict_lengths(self, input_dict: Mapping[str, Union[Sequence[Any], torch.Tensor]]) -> list[int]:
        length = []
        for _, value in input_dict.items():
            length.append(len(value))
        return length

    @property
    def shape(self) -> torch.Size:
        return self.subgoals.shape

    def __len__(self) -> int:
        return self.num_envs

    def __getitem__(self) -> torch.Tensor: ...

    def reset(
        self,
        objects_pose: Mapping[str, torch.Tensor],
    ) -> None:
        """compute subgoal pose by current object pose"""
        subgoals_dict = compute_subgoals(objects_pose, self.single_subgoals)
        self.subgoals = 

    def step(self) -> None:
        "compute current subgoals by obs"
        ...


@torch.jit.script
def compute_subgoals(
    objects_pose: Mapping[str, torch.Tensor], single_subgoals: Mapping[str, torch.Tensor]
) -> dict[str, list[torch.Tensor]]:
    subgoals_dict = defaultdict(list)
    for key, val in single_subgoals.items():
        object_pose = objects_pose[key]
        subgoals_dict[key].append(
            torch.cat(
                combine_frame_transforms(
                    t01=object_pose[:, :3],
                    q01=object_pose[:, 3:7],
                    t12=val[:3].repeat(object_pose.shape[0], 1),
                    q12=val[3:7].repeat(object_pose.shape[0], 1),
                ),
                dim=-1,
            )
        )

    return subgoals_dict
