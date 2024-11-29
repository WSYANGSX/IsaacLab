import torch

a = torch.tensor([0, 0, 0, 0, 0])
b = torch.tensor([1,0,1,0,1],dtype=torch.bool)
print(sum(b))
