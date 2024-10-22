import torch

a = torch.tensor([[1, 2, 3], [1, 2, 3]], device="cuda:0", dtype=torch.float32)
c = torch.clamp(a, min=1, max=2)
print(c)