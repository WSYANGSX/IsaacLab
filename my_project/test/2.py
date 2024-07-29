import torch
from torch import nn

model = nn.Sequential()

input = torch.tensor([1, 2, 3])
output = model(input)

print(output)
