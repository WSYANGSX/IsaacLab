import torch

a = torch.tensor([0, 0, 1, 1, 1])
b = torch.tensor([1, 1, 1, 1, 1])
c = iter(a)

for i in c:
    print(i)
