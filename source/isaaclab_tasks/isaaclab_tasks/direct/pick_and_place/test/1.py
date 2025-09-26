import torch

# a = torch.tensor([1, 2, 3, 4, 5, 6])
# b = torch.tensor([1, 0, 5, 0, 6, 8])
# c = torch.tensor([5, 6, 7, 8, 9, 4])
# print(id(a), id(b), id(c))

# b[:] = a
# a[:] = c

# c[2] = 100

# print(a, b, c)
# print(id(a), id(b), id(c))
# print(a.data_ptr(), c.data_ptr())

# d = c[:]
# print(d, id(d))
# print(d.data_ptr())

# print(a[slice(None, None, None)])

# a = 5
# b = False

# if (a > 0) & (b is False):
#     print(1)

# a = torch.tensor([[1, 2], [3, 4], [5, 6]])
# a[0] = 0
# print(a)

a = torch.tensor([1, 2, 3, 4, 5, 6])
print(a.view(-1, 1))
