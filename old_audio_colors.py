import sys
import pygame
import pygame.surfarray as surfarray
import numpy as np

import pyaudio
import wave
import time

# maybe restructure as having each visual be a class that you pass t and screen to
# transisitons could be classes where the current and next visual are passed and simultaneously calculated
# test if getting and setting the pixel array is faster than swapping out whole array with blit...

# fast way to interpolate data when this ends up on a larger screen

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

def fifo_append(li,obj,L=20):
    if len(li) == L:
        li.pop(0)
    li.append(obj)

class StreamManager(object):
    def __init__(self,data_format,channels,rate,chunk):
        self.data_format = data_format
        self.channels    = channels
        self.rate        = rate
        self.chunk       = chunk
        self.window      = np.blackman(chunk)

    def start(self):
        self.p      = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.data_format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
    
    def ambient_filter(self,data):
        df = np.fft.fft(data)
        return np.fft.ifft(df)
    
    def read(self):
        return self.stream.read(self.chunk,exception_on_overflow=False)
    
    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        

class Visual(object):
    def __init__(self,screen,stream):
        self.screen = screen
        self.stream = stream
        self.width  = screen.get_width()
        self.height = screen.get_height()

class BasicVisual(Visual):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.R       = np.sqrt(width**2 + height**2)
        self.idx_map = np.array([
             [int(CHUNK*np.sqrt((y - width//2)**2 + (x - height//2)**2)/R)
              for x in range(height)] 
              for y in range(width)]) 
        self.pixels    = np.ones((width,height,3))
        self.width_pts = list(range(width))
        
    def iterate(self,t):
        data = self.stream_manager.read()
        data = process_stream_data(data)
        
        data_w = data*window
        data   = to_rgb(data)
        data_b = data[idx_map].reshape(idx_map.shape)
        data_r = data[::-1][idx_map].reshape(idx_map.shape)
        data_f = np.roll(data,len(data)//2)
        data_g = data_f[idx_map].reshape(idx_map.shape)
        
        pixels[:,:,0] = data_r
        pixels[:,:,1] = data_g
        pixels[:,:,2] = data_b
        surfarray.blit_array(self.screen, pixels)
        
        pix = tuple(255*(1+np.sin(f*t))//2 for f in (fr,fg,fb))
        xy = [(w,int(height*(1 + data_w[w])/2)) for w in range(width)]
        xy = np.array([width_pts,data]).T
        xy[:,1] = np.round(.8*height + height*(1 + xy[:,1])/2/8)
        pygame.draw.lines(self.screen,pix,False,xy,2)

def run_visuals(stream_manager):
    vis1 = BasicVisual(screen,stream_manager

    t = 0
    pixels = np.ones((width,height,3))
    R = np.sqrt(width**2 + height**2)
    idx_map = np.array([[int(CHUNK*np.sqrt((y - width//2)**2 + (x - height//2)**2)/R) for x in range(height)] for y in range(width)]) 
    fr,fg,fb = .01,.05,.2
    pygame.init()
    screen = pygame.display.set_mode(size)
    transition = 0
    visual = 1
    
    # for visual = 1, waterfall lines
    lines     = []
    L         = 10
    width_pts = list(range(width))
    offset    = -height/L
        
    while 1:
        print(t)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        if not transition:
            if visual == 0:
                data = stream_manager.read()
                data = process_stream_data(data)
                
                data_w = data*window
                data   = to_rgb(data)
                data_b = data[idx_map].reshape(idx_map.shape)
                data_r = data[::-1][idx_map].reshape(idx_map.shape)
                data_f = np.roll(data,len(data)//2)
                data_g = data_f[idx_map].reshape(idx_map.shape)
                
                pixels[:,:,0] = data_r
                pixels[:,:,1] = data_g
                pixels[:,:,2] = data_b
                surfarray.blit_array(screen, pixels)
                
                pix = tuple(255*(1+np.sin(f*t))//2 for f in (fr,fg,fb))
                xy = [(w,int(height*(1 + data_w[w])/2)) for w in range(width)]
                pygame.draw.lines(screen,pix,False,xy,2) 
            elif visual == 1:
                surfarray.blit_array(screen, BLACK)
                
                data = stream_manager.read()
                data = process_stream_data(data)
                data = data*window
                
                xy = np.array([width_pts,data]).T
                xy[:,1] = np.round(.8*height + height*(1 + xy[:,1])/2/8)
                
                fifo_append(lines,xy,L=L)
                
                pix = tuple(255*(1+np.sin(f*t))//2 for f in (fr,fg,fb))
                
                for i,line in enumerate(lines):
                    _pix = tuple(int(max(10,(i/L)*j)) for j in pix)
                    pygame.draw.lines(screen,_pix,False,line,2)
                
                for line in lines:
                    line[:,1] += offset
                
            if t > 1000:
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

size = width, height = 1024, 500

BLACK = np.zeros((width,height,3))

FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 44100
CHUNK    = 1024

sm = StreamManager(FORMAT,CHANNELS,RATE,CHUNK)
sm.start()

try:
    run_visuals(sm)
except KeyboardInterrupt:
    print('User interruption, shutting down stream...')
    pass

sm.stop()
