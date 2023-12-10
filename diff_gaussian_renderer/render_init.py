import torch
import typing
from diff_gaussian_renderer import DiffGaussianRenderer

def render_gaussians(image_height: int, image_width: int,
                     camera_x: float, camera_y: float, camera_z: float):
    return DiffGaussianRenderer.render_gaussians(image_height, image_width, camera_x, camera_y, camera_z)