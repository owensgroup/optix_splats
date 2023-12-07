import torch
import saxpy
from saxpy import saxpy_cuda, saxpy_optix

x = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda')
y = torch.tensor([4, 5, 6], dtype=torch.float32, device='cuda')
a = 2.0

print('Testing saxpy_cuda')
print(saxpy_cuda(a, x, y))



print('Testing saxpy_optix')
print(saxpy_optix(a, x, y))