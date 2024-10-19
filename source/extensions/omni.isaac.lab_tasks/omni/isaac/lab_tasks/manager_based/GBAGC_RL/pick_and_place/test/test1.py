import torch
from local_projects.utils.math import combine_frame_transforms

# 假设这是两个不同的张量
tensorA = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensorB = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

# 使用torch.stack沿着一个新的维度（默认是0）堆叠它们
# 注意：这将创建一个新的维度，所以结果张量的形状将是(2, 2, 3)
stacked_tensor = torch.hstack((tensorA, tensorB))

b = stacked_tensor.unsqueeze_(dim=0)
c = b.view((2, -1, 3))
d = torch.reshape(b, (2, -1, 3))
print(c)
print(d)