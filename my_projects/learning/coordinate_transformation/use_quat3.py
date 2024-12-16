import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


axis = torch.tensor([[1.0, 1.0, 1.0]])          
angle1 = torch.tensor([torch.pi])
quat1 = quat_from_angle_axis(angle1, axis)

angle2 = torch.tensor([torch.pi / 4])
quat2 = quat_from_angle_axis(angle2, axis)

# same rot axis
print(quat_mul(quat1, quat2))
print(quat_mul(quat2, quat1))


axis1 = torch.tensor([[1.0, 1.0, 1.0]])
angle1 = torch.tensor([torch.pi])
quat1 = quat_from_angle_axis(angle1, axis)

axis2 = torch.tensor([[0.0, 1.0, 1.0]])
angle2 = torch.tensor([torch.pi / 4])
quat2 = quat_from_angle_axis(angle2, axis2)

# different rot axis
print(quat_mul(quat1, quat2))
print(quat_mul(quat2, quat1))
