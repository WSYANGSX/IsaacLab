import torch

a = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float)
b = torch.tensor([[2, 5, 3, 4], [1, 6, 3, 4]], dtype=torch.float)
c = torch.norm(a - b)
print(c)
