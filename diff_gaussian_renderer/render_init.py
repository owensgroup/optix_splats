import torch
import typing
from diff_gaussian_renderer import DiffGaussianRenderer

def render_gaussians(image_height: int, image_width: int):
    return DiffGaussianRenderer.render_gaussians(image_height, image_width)