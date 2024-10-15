import torch

a = torch.tensor([[1, 2, 3], [1, 2, 3]], device="cuda:0", dtype=torch.bool)
b = torch.tensor([[1, 2, 3], [1, 2, 3]], device="cuda:0", dtype=torch.float32)
a = torch.where(b >= 2, torch.ones_like(a), torch.zeros_like(a))
print(a.logical_not())
