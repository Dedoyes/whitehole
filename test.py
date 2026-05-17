import torch

x = torch.randn (5)
print (x.shape)
x = x.unsqueeze (-1)
print (x.shape)
