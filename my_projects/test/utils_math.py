import torch
from omni.isaac.lab.utils.math import *

a = torch.tensor([[0.2, -0.2, 0.3, 0.5], [0.2, -0.2, 0.3, 0.5], [0.2, -0.2, 0.3, 0.5]])
lower = torch.tensor([-100, -100, -100, -100])
upper = torch.tensor([100, 100, 100, 100])

c = unscale_transform(a, lower, upper)
print(c)

a = torch.tensor([0.2, -0.2, 0.3, 0.5])
b = torch.tensor([0.2, -0.2, -0.3, -0.5])
c = torch.tensor([0.2, -0.2, 0.7, 0.8])

print(torch.stack([a, b, c], dim=-1))

a = torch.tensor([True, False, True, True, False])
b = torch.tensor([True, False, True, False, False])
print(torch.where(a | b))
