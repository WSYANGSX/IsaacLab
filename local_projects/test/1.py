import torch
import copy

# 切片相当于浅复制
a = torch.tensor((1, 2, 3))
b = a[:]
b[1] = 5
print(b)
print(a)

a = [1, 2, [1, 2, 3]]
c = a[:]
print(id(a))
print(id(c))
print(id(c[2]))
print(id(a[2]))

# clone
a = torch.tensor((1, 2, 3))
b = a.clone()
print(id(a), id(b))
print(id(a[1]), id(b[1]))  # 对于张量中的元素（在PyTorch中通常是数值类型，如整数或浮点数），这些元素是不可变的，并且没有独立的内存地址来代表单个元素（在C++底层实现中，它们可能是数组或连续内存块中的一部分）
