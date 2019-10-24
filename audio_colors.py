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
CHUNK    = width
window   = np.blackman(CHUNK)

def to_rgb(xyz):
    return (400*(xyz+1)/2)//1

def to_xyz(rgb):
    return 1000*(rgb/256) - 1

def process_stream_data(data):
    waveData = wave.struct.unpack("%dh"%(CHUNK),data)
    waveData = np.array(waveData)
    indata   = waveData
    norm     = np.max(np.abs(indata))
    return indata/norm

pygame.init()


pixels = np.ones((width,height,3))
screen = pygame.display.set_mode(size)#,flags=pygame.OPENGL)

def FIFO(li,item,L=200):
    if len(li) == L:
        li.pop(0)
    li.append(item)
    return li

R = np.sqrt(width**2 + height**2)

idx_map = np.array([[int(CHUNK*np.sqrt((y - width//2)**2 + (x - height//2)**2)/R) for x in range(height)] for y in range(width)]) 


def run_visuals(stream):
    t = 0
    fr,fg,fb = 1113,2017,5012
    t_b = time.time()
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        data = stream.read(CHUNK,exception_on_overflow=False)
        data = process_stream_data(data)
        
        data_w = data*window
        data_b = data[idx_map].reshape(idx_map.shape)
        
        pixels[:,:,:2] = 0
        pixels[:,:,2] = data_b 
        
        t += 1
        pix = tuple(255*(1+np.sin(f*t))//2 for f in (fr,fg,fb))
        xy = [(w,int(height*(1 + data_w[w])/2)) for w in range(width)]
        pygame.draw.lines(screen,pix,xy) 
        surfarray.blit_array(screen, pixels)
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
