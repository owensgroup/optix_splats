import torch
from saxpy import saxpy_ext

def saxpy_cuda(a, x, y):
    return saxpy_ext.saxpy(x, y, a)

def saxpy_optix(a, x, y):
    return saxpy_ext.saxpy_optix(x, y, a)