import torch
from .DiffGaussianRenderer import render_gaussians as _render_gaussians
from .DiffGaussianRenderer import OptixState

__all__ = ['OptixState']

def render_gaussians(camera_x: float, camera_y: float, camera_z: float,
                     lookat_x: float, lookat_y: float, lookat_z: float,
                     up_x: float, up_y: float, up_z: float, state: OptixState):
    return _render_gaussians(camera_x, camera_y, camera_z, lookat_x, lookat_y, lookat_z, up_x, up_y, up_z, state)