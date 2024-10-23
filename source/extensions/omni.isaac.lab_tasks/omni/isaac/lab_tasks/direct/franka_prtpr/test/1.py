import torch

a = torch.tensor([[1, 2, 3], [1, 2, 3]], device="cuda:0", dtype=torch.float32)
b = torch.zeros_like(a, dtype=torch.bool)
print(a.device, b)
