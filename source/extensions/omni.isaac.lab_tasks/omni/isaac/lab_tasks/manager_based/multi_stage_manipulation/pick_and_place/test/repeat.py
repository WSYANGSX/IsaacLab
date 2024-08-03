import torch

a = torch.tensor([True, False, True])
b = torch.full_like(a, fill_value=50, dtype=torch.float32)
print(b)