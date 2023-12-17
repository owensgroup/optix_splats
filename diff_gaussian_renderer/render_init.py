import torch
import typing
from diff_gaussian_renderer import DiffGaussianRenderer

def render_gaussians(image_height: int, image_width: int,
                     camera_x: float, camera_y: float, camera_z: float,
                     lookat_x: float, lookat_y: float, lookat_z: float,
                     up_x: float, up_y: float, up_z: float, state: DiffGaussianRenderer.State):
    return DiffGaussianRenderer.render_gaussians(image_height, image_width, camera_x, camera_y, camera_z, lookat_x, lookat_y, lookat_z, up_x, up_y, up_z, state)

def create_state() -> DiffGaussianRenderer.State:
    return DiffGaussianRenderer.createState()