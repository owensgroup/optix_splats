import torch
import diff_gaussian_renderer
from diff_gaussian_renderer import render_gaussians
import matplotlib.pyplot as plt

def show_image(image_width, image_height, camera_x, camera_y, camera_z):
    a = render_gaussians(image_height, image_width, camera_x, camera_y, camera_z)
    a = a.to('cpu')
    r = ((a >> 24) & 0xFF).byte()
    g = ((a >> 16) & 0xFF).byte()
    b = ((a >> 8) & 0xFF).byte()
    alpha = (a & 0xFF).byte()

    a = torch.stack([r, g, b, alpha], dim=-1)

    # a BGR -> RGB
    a = a[..., [2, 1, 0, 3]]


    # plot a
    plt.imshow(a)
    plt.show()

show_image(2560, 1440, 0.0, 0.0, 3.0)
print(a.shape)