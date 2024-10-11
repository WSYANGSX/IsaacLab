import torch
import numpy as np
import math

a = torch.inf
b = np.inf
c = math.inf

print(a==b)
print(b==c)
print(a==c)
print(a==inf)