import pygame
import time

t = 0
pygame.init()
screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)

colors = [(255,0,0),
          (0,255,0),
          (0,0,255)]

running = True
while running:
    print(t)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type is pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
    
    screen.fill(colors[t])
    pygame.display.flip()

    time.sleep(2)

    t += 1
    if t >= len(colors):
        running = False
print("shutting it down")
