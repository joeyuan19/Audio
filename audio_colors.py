import sys
import pygame
import pygame.surfarray as surfarray
import numpy as np

import pyaudio
import wave
import time

FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 44100
CHUNK    = 1024
window   = np.blackman(CHUNK)

def to_rgb(xyz):
    return (256*(xyz+1)/2)//1

def to_xyz(rgb):
    return 1000*(rgb/256) - 1

def process_stream_data(data):
    waveData = wave.struct.unpack("%dh"%(CHUNK),data)
    waveData = np.array(waveData)
    indata   = waveData*window
    return np.average(indata)

pygame.init()

size = width, height = 500, 500

pixels = np.zeros((height,width,3))
screen = pygame.display.set_mode(size)

def FIFO(li,item,L=50):
    if len(li) == 50:
        li.pop(0)
    li.append(item)
    return li


def run_visuals(stream):
    recent_values = []
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        
        data = stream.read(CHUNK,exception_on_overflow=False)
        a = process_stream_data(data)
        recent_values = FIFO(recent_values,a)
        low   = min(recent_values)
        high  = max(recent_values)
        
        p = (a-low)/(high-low)

        pixels[:,:,0] = (256*p)//1
        
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

stream.stop_stream()
stream.close()
p.terminate()
