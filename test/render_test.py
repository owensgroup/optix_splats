import torch
from diff_gaussian_renderer import render_gaussians, OptixState
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from plyfile import PlyData

import numpy as np
import sys

import math
import pygame
import time

print("HELLO")

pygame.init()

from PIL import Image

# import matplotlib
# print(matplotlib.get_backend())

width = 2560
height = 1440

font = pygame.font.SysFont('Arial', 25)
clock = pygame.time.Clock()

def pil_image_to_pygame(pil_image):
    # Ensure the image is in the correct format
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')
    
    # Get the size of the image
    width, height = pil_image.size
    
    # Get the raw bytes of the image
    image_string = pil_image.tobytes("raw", "RGBA")
    
    # Create a Pygame surface from the string buffer
    return pygame.image.fromstring(image_string, (width, height), 'RGBA')

state = None
def get_image(camera_position,
              lookat, up, state):
    
    # if state is None:
    #     print("CREATING STATE")
    #     state = create_state()
    render_start_time = time.time()
    a = render_gaussians(camera_position.x, camera_position.y, camera_position.z, lookat.x, lookat.y, lookat.z, up.x, up.y, up.z, state)
    r = ((a >> 24) & 0xFF).byte()
    g = ((a >> 16) & 0xFF).byte()
    b = ((a >> 8) & 0xFF).byte()
    alpha = (a & 0xFF).byte()
    a = torch.stack([r, g, b, alpha], dim=-1)

    # a BGR -> RGB
    a = a[..., [2, 1, 0, 3]]

    render_end_time = time.time()
    print("RENDER TIME: ", render_end_time - render_start_time)

    a = a.to('cpu')
    # get the first 3 channels of a
    a = Image.fromarray(a.numpy())
    # Convert to Image
    return pil_image_to_pygame(a)

class TrackballCamera:
    def __init__(self):
        self.azimuth = 0  # Rotation around the y-axis
        self.elevation = 0  # Rotation around the x-axis
        self.distance = 5  # Distance from the target
        self.target = pygame.math.Vector3(0, 0, 0)  # The point to look at
    

    def handle_input(self, event):
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:  # Left mouse button is down
                x, y = event.rel  # Relative motion
                self.azimuth += x * 0.5  # Sensitivity factor
                self.elevation -= y * 0.5  # Sensitivity factor
                self.elevation = max(-89, min(89, self.elevation))  # Clamp

    def get_position(self):
        # Convert angles from degrees to radians
        azimuth_rad = math.radians(self.azimuth)
        elevation_rad = math.radians(self.elevation)

        # Calculate camera position
        x = self.distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        y = self.distance * math.sin(elevation_rad)
        z = self.distance * math.cos(elevation_rad) * math.cos(azimuth_rad)

        # Translate based on the target position
        camera_position = self.target + pygame.math.Vector3(x, y, z)

        return camera_position

camera = TrackballCamera()

lookat = pygame.math.Vector3(0, 0, 0)
up = pygame.math.Vector3(0, 1, 0)
camera_position = camera.get_position()



window = pygame.display.set_mode((width, height))

# Set a title for the window
pygame.display.set_caption('Pygame Window')
# Main loop
running = True
i = 0
ply_path = 'test/example.ply'

ply_data = PlyData.read(ply_path)
# Assuming 'ply_data' is already read from a PLY file and contains elements
first_element = ply_data.elements[0]
properties_lists = {prop.name: np.array(first_element.data[prop.name]) for prop in first_element.properties}
print(properties_lists.keys())
x = properties_lists['x']
y = properties_lists['y']
z = properties_lists['z']

opacity = properties_lists['opacity']

scale_x = properties_lists['scale_0']
scale_y = properties_lists['scale_1']
scale_z = properties_lists['scale_2']

a = properties_lists['rot_0']
b = properties_lists['rot_1']
c = properties_lists['rot_2']
d = properties_lists['rot_3']

means = torch.from_numpy(np.column_stack((x, y, z)))
scales = torch.from_numpy(np.column_stack((scale_x, scale_y, scale_z)))
rotations = torch.from_numpy(np.column_stack((a, b, c, d)))

state = OptixState(width, height)
state.build_ias(means, scales, rotations)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        camera.handle_input(event)
    get_image_start = time.time()
    a = get_image(camera.get_position(), lookat, up, state)
    get_image_end = time.time()

    window.fill((0, 0, 0))
    window.blit(a, (0, 0))

    fps = int(clock.get_fps())
    fps_text = font.render(f'FPS: {fps}', True, pygame.Color('white'))
    window.blit(fps_text, (10, 10))  # Position the FPS text at the top-left corner

    # Update the display
    pygame.display.flip()

    clock.tick()
    

# Quit Pygame
pygame.quit()
