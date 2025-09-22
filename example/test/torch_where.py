import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.0, 1.0, 4.0])
direction_large_change = torch.where(
    a >= b,
    torch.ones_like(a),
    torch.zeros_like(a),
)
print(direction_large_change)