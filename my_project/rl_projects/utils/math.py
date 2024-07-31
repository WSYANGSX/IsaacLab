import torch

from omni.isaac.lab.utils.math import quat_mul, quat_conjugate


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    """compute rad diff between two quat."""
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    )  # changed quat convention
