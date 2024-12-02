import torch
import torch.nn as nn


# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义模型层
        self.fc1 = nn.Linear(7, 256)
        self.bn1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.LayerNorm(64)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.LayerNorm(32)

        self.fc5 = nn.Linear(32, 3)

    def forward(self, x):
        # 定义前向传播
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


model = torch.load(
    "/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/inverse_kinematic/v1/forward_model.pth"
)

a = torch.tensor(
    [
        [
            1.431610107421875, -0.5103200674057007, 2.570969581604004, -1.4957352876663208, -2.7472922801971436, 0.6914240121841431, 1.9748085737228394
        ],
    ],
    device="cuda",
)

b = torch.tensor(
    [[0.5265768766403198, 0.0371532067656517, 0.3329574465751648, 0.7813296318054199]],
    device="cuda",
)


out = model(a)
print(out)


"""Helper Function"""


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise."""
    mag = torch.tensor(mag, device=other.device, dtype=torch.float32).repeat(
        other.shape[0]
    )
    return torch.abs(mag) * torch.sign(other)


@torch.jit.script
def euler_xyz_from_quat(
    quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to Euler angles in radians."""
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(
        torch.abs(sin_pitch) >= 1,
        copysign(torch.pi / 2.0, sin_pitch),
        torch.asin(sin_pitch),
    )

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    return (
        roll % (2 * torch.pi),
        pitch % (2 * torch.pi),
        yaw % (2 * torch.pi),
    )
