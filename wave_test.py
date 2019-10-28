import matplotlib.pyplot as plt
import numpy as np
import time

def radius(x,y):
    return np.sqrt(x**2 + y**2)

def wave(x,y,a,f):
    r = radius(x,y)
    return np.exp(-a*r)*np.cos(f*r)

def wave_mask(width,height,a,f):
    x,y = np.mgrid[-(width//2):width//2:1j*width, -(height//2):height//2:1j*height]
    return wave(x,y,a,f) 

x = np.linspace(0,100,1000)
t = 0

for t in range(1,100):
    w = wave_mask(500,500,1/t/20,t/1000)
    plt.cla()
    plt.imshow(w)
    plt.pause(0.01)
