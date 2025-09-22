import torch

a = torch.tensor([[0], [1], [2], [1], [0]])
b = torch.tensor([[1, 2, 3, 4], [0, 5, 4, 7], [1, 6, 4, 5], [0, 0, 0, 0], [6, 7, 1, 5]])
c = torch.tensor([[0, 0, 0, 0], [0, 5, 4, 7], [1, 6, 4, 5], [0, 0, 0, 0], [0, 0, 0, 0]])
b = torch.where(a < 1, c, b)
print(b)
