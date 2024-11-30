import torch


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_unique(q: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form where the real part is non-negative.

    Quaternion representations have a singularity since ``q`` and ``-q`` represent the same
    rotation. This function ensures the real part of the quaternion is non-negative.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Standardized quaternions. Shape is (..., 4).
    """
    return torch.where(q[..., 0:1] < 0, -q, q)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)


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
    theta = torch.arccos(torch.clamp(torch.sum(u1 * u2, dim=-1, keepdim=True), min=-1.0, max=1.0))

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
def rotation_distance(object_rot: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    """compute rad diff between two quat."""
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


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

    v1_zero_mask = (v1 == 0).all(dim=-1, keepdim=True)
    v2_zero_mask = (v2 == 0).all(dim=-1, keepdim=True)
    zero_mask = torch.where(v1_zero_mask | v2_zero_mask, 0, 1)

    return theta_radians * zero_mask


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor | None = None, q12: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t12: Position of frame 2 w.r.t. frame 1. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q12: Quaternion orientation of frame 2 w.r.t. frame 1 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 0.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    # compute translation
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)
