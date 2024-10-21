import torch

a = slice(1, 5, 2)
b = torch.arange(6)
print(b[a])