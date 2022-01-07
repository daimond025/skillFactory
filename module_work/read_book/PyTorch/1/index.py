import torch

d = torch.rand(5,5)
print(d.type())
print(d.to(dtype=torch.int))
print(d)