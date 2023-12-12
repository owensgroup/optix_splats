import torch
import diff_gaussian_renderer
from diff_gaussian_renderer import render_gaussians
import matplotlib.pyplot as plt

import numpy as np
import sys

import pygame

print("HELLO")

pygame.init()

from PIL import Image

# import matplotlib
# print(matplotlib.get_backend())

width = 2560
height = 1440

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

def get_image(image_width, image_height, camera_x, camera_y, camera_z):
    a = render_gaussians(image_height, image_width, camera_x, camera_y, camera_z)
    a = a.to('cpu')
    r = ((a >> 24) & 0xFF).byte()
    g = ((a >> 16) & 0xFF).byte()
    b = ((a >> 8) & 0xFF).byte()
    alpha = (a & 0xFF).byte()

    a = torch.stack([r, g, b, alpha], dim=-1)

    # a BGR -> RGB
    a = a[..., [2, 1, 0, 3]]

    # Convert to Image
    a = Image.fromarray(a.numpy())
    return pil_image_to_pygame(a)
# fig, ax = plt.subplots()

zoom = 3.0
a = get_image(width, height, 0.0, 0.0, zoom)

# Display the image
# im = ax.imshow(a)
# fig.canvas.draw()

# # Define a callback function to update the image
# def on_key(event):
#     global zoom, a
#     print('press', event.key)
#     sys.stdout.flush()
#     # Get the x, y coordinates of the click
#     if event.key == 'up':
#     if event.key == 'down':
#         a = get_image(width, height, 0.0, 0.0, zoom)  # Set the entire image to red
#         im.set_data(a)
#         fig.canvas.draw()
#         print(zoom)
#         sys.stdout.flush()

window = pygame.display.set_mode((width, height))

# Set a title for the window
pygame.display.set_caption('Pygame Window')


print("HELLO")
# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Check which key was pressed
            if event.key == pygame.K_UP:
                zoom -= 0.1
                a = get_image(width, height, 0.0, 0.0, zoom)
            elif event.key == pygame.K_DOWN:
                zoom += 0.1
                a = get_image(width, height, 0.0, 0.0, zoom)
    
    window.fill((0, 0, 0))
    window.blit(a, (0, 0))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()

# Connect the callback function to the click event
# cid = fig.canvas.mpl_connect('key_press_event', on_key)

# plt.show()