import torch

torch.set_printoptions(profile="full")


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
def calculate_angle_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
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


# 测试代码
a = torch.tensor([[-0.0, -0.0, -0.0], [-1.0, -1.0, -1.0]], dtype=torch.float64)
b = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
print(calculate_angle_between_vectors(a, b))  # 应该输出接近 π 的值
