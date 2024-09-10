import torch

a = torch.tensor([1, 2, 3])

print(torch.unsqueeze(a, dim=0).view(-1,  1))
