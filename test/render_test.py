import torch
import diff_gaussian_renderer
from diff_gaussian_renderer import render_gaussians

a = render_gaussians(800, 600)
print(a.shape)