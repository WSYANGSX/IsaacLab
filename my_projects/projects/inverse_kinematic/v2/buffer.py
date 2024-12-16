import torch
import random


class Buffer(object):
    def __init__(self, herizon_length):
        """Create buffer.
        Parameters
        ----------
        herizon_length: int
            Max number of data to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._current_arm_pos = torch.zeros(
            (herizon_length, 7), dtype=torch.float32, device="cuda:0"
        )
        self._target_arm_pos = torch.zeros(
            (herizon_length, 7), dtype=torch.float32, device="cuda:0"
        )
        self._target_pos = torch.zeros(
            (herizon_length, 3), dtype=torch.float32, device="cuda:0"
        )
        self._target_rot = torch.zeros(
            (herizon_length, 4), dtype=torch.float32, device="cuda:0"
        )

        self._maxsize = herizon_length
        self._next_idx = 0
        self._curr_size = 0

    def __len__(self):
        return self._curr_size

    def add(self, current_arm_pos, target_arm_pos, target_pos, target_rot):
        self._curr_size = min(self._curr_size + 1, self._maxsize)

        self._current_arm_pos[self._next_idx] = torch.tensor(current_arm_pos)
        self._target_arm_pos[self._next_idx] = torch.tensor(target_arm_pos)
        self._target_pos[self._next_idx] = torch.tensor(target_pos)
        self._target_rot[self._next_idx] = torch.tensor(target_rot)

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _get(self, idx):
        return (
            self._current_arm_pos[idx],
            self._target_arm_pos[idx],
            self._target_pos[idx],
            self._target_rot[idx],
        )

    def _encode_sample(self, idxes):
        return (
            self._current_arm_pos[idxes],
            self._target_arm_pos[idxes],
            self._target_pos[idxes],
            self._target_rot[idxes],
        )

    def sample(self, batch_size):
        idxes = [random.randint(0, self._curr_size - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)