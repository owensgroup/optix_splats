import pygame


def draw_whitebuffer(screen):
    white = (255, 255, 255)
    screen.fill(white)

width = 2560
height = 1440
pygame.init()
size = (width, height)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Whitebuffer")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_whitebuffer(screen)
    pygame.display.flip()

pygame.quit()

