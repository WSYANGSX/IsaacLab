import torch

a = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
b = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
c = torch.where(a&b, 2, 0)
print(c)
