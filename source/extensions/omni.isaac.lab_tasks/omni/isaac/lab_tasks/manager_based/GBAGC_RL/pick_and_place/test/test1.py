import torch


a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
b = torch.tensor([1.5, 2.6, 4.6, 6.8, 0.5])
succ = (a <= 0.3) & (b >= 2.6)
print(succ)
