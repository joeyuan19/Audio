import sys
import pygame
import pygame.surfarray as surfarray
import numpy as np

import pyaudio
import wave
import time

size = width, height = 1024, 500

FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 44100
CHUNK    = 1024
window   = np.blackman(CHUNK)

def to_rgb(xyz):
    return (255*(xyz+1)/2)//1

def to_xyz(rgb):
    return 1000*(rgb/256) - 1

def process_stream_data(data):
    waveData = wave.struct.unpack("%dh"%(CHUNK),data)
    waveData = np.array(waveData)
    indata   = waveData
    norm     = np.max(np.abs(indata))
    return indata/norm

def radius(x,y):
    return np.sqrt(x**2 + y**2)

def drop(x,y,a,f):
    r = radius(x,y)
    return np.exp(-a*r)*(.5 + .5*np.cos(f*r))

def drop_mask(width,height,a,f):
    x,y,z = np.mgrid[-(width//2):width//2:1j*width, -(height//2):height//2:1j*height,-1:1:3j]
    return wave(x,y,a,f)

def run_visuals(stream):
    t = 0
    pixels = np.ones((width,height,3))
    R = np.sqrt(width**2 + height**2)
    idx_map = np.array([[int(CHUNK*np.sqrt((y - width//2)**2 + (x - height//2)**2)/R) for x in range(height)] for y in range(width)]) 
    fr,fg,fb = .01,.05,.2
    pygame.init()
    screen = pygame.display.set_mode(size)
    transition = 0
    while 1:
        print(t)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        if not transition:
            data = stream.read(CHUNK,exception_on_overflow=False)
            data = process_stream_data(data)
            
            data_w = data*window
            data   = to_rgb(data)
            data_b = data[idx_map].reshape(idx_map.shape)
            data_r = data[::-1][idx_map].reshape(idx_map.shape)
            
            pixels[:,:,0] = data_r
            pixels[:,:,1] = 0
            pixels[:,:,2] = data_b
            surfarray.blit_array(screen, pixels)
            
            pix = tuple(255*(1+np.sin(f*t))//2 for f in (fr,fg,fb))
            xy = [(w,int(height*(1 + data_w[w])/2)) for w in range(width)]
            pygame.draw.lines(screen,pix,False,xy,2) 
            if t > 10:
                transition = np.random.randint(1,2)
                t = 0
            else:
                t += 1
        else:
            pixels = pygame.PixelArray(screen)
            if transition == 1:
                bars = 10
                rows = height//bars
                inc  = 20
                for n in range(bars):
                    if n%2 == 0:
                        pixels[t+inc:,n*rows:(n+1)*rows] = pixels[t:-inc,n*rows:(n+1)*rows] 
                        pixels[:t+inc,n*rows:(n+1)*rows] = (0,0,0)
                    else:
                        pixels[:width-t-inc,n*rows:(n+1)*rows] = pixels[inc:width-t,n*rows:(n+1)*rows] 
                        pixels[width-t:,n*rows:(n+1)*rows] = (0,0,0)
                if t >= width//2:
                    pixels = np.zeros((width,height,3))
                    transition = 0
                    t = 0
                else:
                    t += inc
            if transition == 2:
                inc = 1
                t += inc
                d = drop_mask(width,height,1/(20*t),t/1000)
                if t >= 100:
                    t = 0
                    transition = 0
                    pixels = np.ones((width,height,3))
            
        
        pygame.display.flip()

p = pyaudio.PyAudio()
s = p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
           frames_per_buffer=CHUNK)
try:
    run_visuals(s)
except KeyboardInterrupt:
    print('User interruption, shutting down stream...')
    pass

s.stop_stream()
s.close()
p.terminate()
