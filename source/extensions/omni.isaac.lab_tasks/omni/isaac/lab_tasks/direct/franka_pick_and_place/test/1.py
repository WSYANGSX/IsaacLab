import torch

a = torch.tensor([[3, 3], [1, 2], [4, 5]])
b = torch.tensor([1, 2])
print(id(a))
a[0, 1].zero_()
print(a)
