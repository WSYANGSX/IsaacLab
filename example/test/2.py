import torch

a = torch.tensor([1, 2, 3])
b = a.view(-1, 1)
print(id(a))
print(id(b))
print(a)
print(b)
