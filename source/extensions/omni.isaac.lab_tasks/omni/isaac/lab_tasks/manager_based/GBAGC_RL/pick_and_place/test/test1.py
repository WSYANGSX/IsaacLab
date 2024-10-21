import torch

# 假设这是两个不同的张量
tensorA = torch.tensor(
    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
)
print(tensorA.shape)

curr_indices = torch.tensor([0, 1, 1])
sample_indices = [curr_indices[i] + i * 2 for i in range(len(curr_indices))]
print(sample_indices)
tensorB = tensorA.view(-1, 3)
print(tensorB[sample_indices, :])
