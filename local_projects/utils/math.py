import torch

from omni.isaac.lab.utils.math import normalize, quat_mul, quat_conjugate


def quat_from_vectors(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    # 规范化向量
    u1 = normalize(vector1)
    u2 = normalize(vector2)

    # 计算旋转轴（叉积），确保输入的向量是三维的
    n = torch.cross(u1, u2)

    # 处理 n 为零向量的情况（可选，但有助于避免潜在的除零错误）
    epsilon = 1e-7
    n_norm = torch.sqrt(torch.sum(n**2, dim=-1, keepdim=True))
    n = torch.where(n_norm > epsilon, n / n_norm, torch.zeros_like(n))

    # 计算旋转角
    theta = torch.arccos(
        torch.clamp(torch.sum(u1 * u2, dim=-1, keepdim=True), min=-1.0, max=1.0)
    )

    # 构造四元数
    sin_half_theta = torch.sin(theta / 2)
    # 确保 sin_half_theta 可以与 n 广播
    sin_half_theta = sin_half_theta.unsqueeze(-1)  # 增加一个维度以匹配 n 的形状
    w = torch.cos(theta / 2)
    x, y, z = n * sin_half_theta

    # 将 w, x, y, z 堆叠成四元数张量
    quat = torch.stack([w, x, y, z], dim=-1)

    return quat


@torch.jit.script
def rotation_axis(object_rot: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    """compute rot axis between two quat."""
    rot_diff = normalize(quat_mul(target_rot, quat_conjugate(object_rot)))

    w = rot_diff[:, 0]
    a = rot_diff[:, 1]
    b = rot_diff[:, 2]
    c = rot_diff[:, 3]
    rot_axis = torch.cat(
        [
            torch.reshape(a / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
            torch.reshape(b / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
            torch.reshape(c / torch.sqrt(1 - w**2 + 1e-9), (-1, 1)),
        ],
        dim=-1,
    )

    return rot_axis


@torch.jit.script
def rotation_distance(
    object_rot: torch.Tensor, target_rot: torch.Tensor
) -> torch.Tensor:
    """compute rad diff between two quat."""
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    )  # changed quat convention


@torch.jit.script
def calculate_angle_between_vectors(v1: torch.Tensor, v2: torch.Tensor):
    # 确保输入的张量具有相同的批次大小和向量维度
    assert v1.shape == v2.shape, "v1 and v2 must have the same shape"

    # 计算向量的长度
    norm_v1 = normalize(v1)
    norm_v2 = normalize(v2)

    # 计算点积
    dot_product = torch.sum(norm_v1 * norm_v2, dim=-1, keepdim=True)

    # 计算夹角的余弦值
    cos_theta = torch.clamp(dot_product, -1, 1)

    # 使用 arccos 获取夹角（以弧度为单位）
    theta_radians = torch.arccos(cos_theta)

    return theta_radians
