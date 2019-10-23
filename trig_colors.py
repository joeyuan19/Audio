import sys
import pygame
import pygame.surfarray as surfarray
import numpy as np

def to_rgb(xyz):
    return (256*(xyz+1)/2)//1

def to_xyz(rgb):
    return 1000*(rgb/256) - 1

def trig(arr,freq,phase):
    arr = to_xyz(arr)
    arr[:,:,R] = np.sin(freq[0]*arr[:,:,R]+phase[0])
    arr[:,:,G] = np.sin(freq[1]*arr[:,:,G]+phase[1])
    arr[:,:,B] = np.sin(freq[2]*arr[:,:,B]+phase[2])
    return to_rgb(arr)

pygame.init()

# cycle through a few options or press buttons for different options?

size = width, height = 500, 500
black = 0, 0, 0

R,G,B = 0,1,2

X = np.array([np.linspace(-10,10,width) for i in range(height)])
Y = X.T
base_array = np.zeros((width,height,3))
base_array[:,:,0] = X
base_array[:,:,1] = Y
base_array[:,:,2] = X*Y

phase = np.array((0,0,0))
t = 0
RGB = trig(base_array,(1,1,1),phase)

screen = pygame.display.set_mode(size)
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    surfarray.blit_array(screen, RGB)
    pygame.display.flip()
     
    t += .1
    freq   = (1 + 5*(1+np.sin(t)),.1 + 2*(1+np.cos(t)),1)
    phase = (phase + (.2,.3,.5))%(2*np.pi)
    RGB = trig(base_array,freq,phase)


    
