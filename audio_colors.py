import sys
import pygame
import pygame.surfarray as surfarray
import pygame.gfxdraw

import numpy as np

import pyaudio
import audioop
import wave
import time
import pickle

from multiprocessing.dummy import Pool as ThreadPool

# maybe restructure as having each visual be a class that you pass t and screen to
# transisitons could be classes where the current and next visual are passed and simultaneously calculated
# test if getting and setting the pixel array is faster than swapping out whole array with blit...

# fast way to interpolate data when this ends up on a larger screen

def to_rgb(xyz):
    return (255*(xyz+1)/2)//1

def to_xyz(rgb):
    return 1000*(rgb/256) - 1

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

class Ball(object):
    def __init__(self,pos,rate,r,color):
        self.pos   = np.array(pos,dtype=float)
        self.rate  = np.array(rate,dtype=float)
        self.r     = r
        self.color = color

    def set_color(self,color):
        self.color = color

    def set_rate(self,rate):
        self.rate = rate

    def get_rate(self):
        return self.rate 

    def update(self):
        self.pos += self.rate

class BallManager(object):
    def __init__(self,N,initial_pos,initial_rate,initial_radii,initial_colors,color_freqs,width,height):
        self.N      = N
        self.balls  = [Ball(*args) for args in zip(initial_pos,initial_rate,initial_radii,initial_colors)]
        
        self.color_f = color_freqs
        
        self.N_threads = 4
        
        self.width  = width
        self.height = height
        self.rev_x  = np.array((-1,1))
        self.rev_y  = np.array((1,-1))
        
        self.G = 100

    def update(self,t):
        t0 = time.time()
        r  = [(ball_a.pos+ball_a.rate)-(ball_b.pos+ball_b.rate)
              for i,ball_a in enumerate(self.balls[:-1])
              for j,ball_b in enumerate(self.balls[i+1:])]
        r2 = [np.dot(_r,_r) for _r in r]
        p  = [np.arctan2(_r[1],_r[0]) for _r in r]
        self.gravity(r,r2,self.G)
        self.collisions(r2,p)
        self.walls()
        self.color_update(t)
        [ball.update() for ball in self.balls]
        print(time.time()-t0)

    def collisions(self,mag,phase):
        k = 0
        for i,ball_a in enumerate(self.balls[:-1]):    
            for j,ball_b in enumerate(self.balls[i+1:]):    
                r = mag[k]
                p = phase[k]
                if r <= (ball_a.r + ball_b.r)**2:
                     self.reflect_rate(ball_a,p)
                     self.reflect_rate(ball_b,p)
                k += 1

    def collisions_threaded(self):
        with ThreadPool(self.N_threads) as pool:
            results = pool.map(self.calc_collision, [(i,j) for i in range(self.N-1) for j in range(i+1,self.N)])

    def calc_collision(self,*args):
        i,j = tuple(*args)
        ball_a = self.balls[i]
        ball_b = self.balls[j]
        dr = (ball_a.pos+ball_a.rate)-(ball_b.pos+ball_b.rate)
        r  = np.sqrt(np.dot(dr,dr))
        p  = np.arctan2(dr[1],dr[0])
        if r <= ball_a.r + ball_b.r:
             self.reflect_rate(ball_a,p)
             self.reflect_rate(ball_b,p)

    def gravity(self,r,r2,G):
        acc = np.zeros((self.N,2))
        k = 0
        for i,ball_a in enumerate(self.balls[:-1]):
            for j,ball_b in enumerate(self.balls[i+1:]):
                dr = r[k]
                m  = r2[k]
                acc[i,:] += -dr*G/m
                acc[i+1+j,:] += dr*G/m
                k += 1
        [ball.set_rate(ball.rate+acc[i]) for i,ball in enumerate(self.balls)]
    
    def gravity_threaded(self):
        with ThreadPool(self.N_threads) as pool:
            pool.map(self.calc_gravity, np.arange(self.N))

    def calc_gravity(self,i):
        acc = np.array((0.,0.))
        for j,ball_b in enumerate(self.balls):
            if i != j:
                dr = ball_b.pos - self.balls[i].pos
                acc += -self.G/np.dot(dr,dr)
        self.balls[i].set_rate(self.balls[i].rate+acc)
    
    def rotation(self,vector,angle):
        c,s = np.cos(angle),np.sin(angle)
        return np.dot(vector,np.array([[c,-s],[s,c]]))

    def reflect_rate(self,ball,angle):
        ball.set_rate(self.rotation(self.rev_x*self.rotation(ball.rate,angle),-angle))

    def walls(self):
        [self.reflect_wall(ball) for ball in self.balls]

    def reflect_wall(self,ball):
        x,y = (ball.pos+ball.rate)
        if x - ball.r <= 0 or x + ball.r >= self.width:
            ball.set_rate(self.rev_x*ball.rate)
        elif y - ball.r <= 0 or y + ball.r >= self.height:
            ball.set_rate(self.rev_y*ball.rate)
    
    def color_update(self,t):
        colors = (255*(np.sin(self.color_f*t)+1)/2).astype(int)
        for ball,color in zip(self.balls,colors):
            ball.set_color(color)
    
    def draw_ball(self,surface,ball):
        x,y = map(int,ball.pos)
        pygame.gfxdraw.filled_circle(surface,x,y,ball.r,ball.color)
    
    def draw_all(self,surface):
        for ball in self.balls:
            self.draw_ball(surface,ball)

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
    
    def process_stream_data(self,data):
        waveData = wave.struct.unpack("%dh"%(CHUNK),data)
        return np.array(waveData)

    def normalize(self,data):
        norm = np.max(np.abs(data))
        return data/norm

    def ambient_filter(self,data):
        df = np.fft.fft(data)
        return np.fft.ifft(df)
    
    def _read(self):
        return self.stream.read(self.chunk,exception_on_overflow=False)
        
    def read(self):
        data = self.process_stream_data(self._read())
        return data
    
    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
    
    def volume(self):
        return audioop.rms(self._read(),2)

class Visual(object):
    def __init__(self,screen,stream):
        self.screen = screen
        self.stream = stream
        self.width  = screen.get_width()
        self.height = screen.get_height()

class BasicVisual(Visual):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.R       = np.sqrt(self.width**2 + self.height**2)
        self.idx_map = np.array([
             [int(CHUNK*np.sqrt((y - self.width//2)**2 + (x - self.height//2)**2)/self.R)
              for x in range(height)] 
              for y in range(width)]) 
        self.pixels    = np.ones((width,height,3))
        self.width_pts = np.linspace(0,width,self.stream.chunk)
        self.color_f   = .01,.05,.5
        self.chan_f    = np.array((1/10,1/20,1/50))
        
    def iterate(self,t):
        data = self.stream.read()
        data = self.stream.normalize(data)
        
        data_w = data*self.stream.window
        data   = to_rgb(data)
        data_b = data[self.idx_map].reshape(self.idx_map.shape)
        data_r = data[::-1][self.idx_map].reshape(self.idx_map.shape)
        data_f = np.roll(data,len(data)//2)
        data_g = data_f[self.idx_map].reshape(self.idx_map.shape)
        
        amp = np.sin(self.chan_f*t)**2
        self.pixels[:,:,0] = data_r*amp[0]
        self.pixels[:,:,1] = data_g*amp[1]
        self.pixels[:,:,2] = data_b*amp[2]
        print(self.width,self.screen.get_width())
        surfarray.blit_array(self.screen, self.pixels)
        
        pix = tuple(255*(1+np.sin(f*t))//2 for f in self.color_f)
        xy = np.array([self.width_pts,data_w]).T
        xy[:,1] = np.round(self.height*(1 + xy[:,1])/2)
        pygame.draw.lines(self.screen,pix,False,xy,2)

class JoyDivisionVisual(Visual):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lines     = []
        self.L         = 12
        self.width_pts = np.linspace(0,width,self.stream.chunk)
        self.offset    = -self.height/self.L
        self.color_f   = .3,.5,.1
        
    def iterate(self,t):
        self.screen.fill((0,0,0))
        
        data = self.stream.read()
        data = self.stream.normalize(data)
        data = data*self.stream.window
        
        xy = np.array([self.width_pts,data]).T
        xy[:,1] = np.round(.85*self.height + self.height*(1 + xy[:,1])/2/8)
        
        pix = tuple(50 + 205*(1+np.sin(f*t))//2 for f in self.color_f)
        fifo_append(self.lines,(xy,pix),L=self.L)
        
        for i,line_info in enumerate(self.lines):
            line,_pix = line_info
            _pix = tuple(int(max(10,(i/self.L)*j)) for j in _pix)
            pygame.draw.lines(self.screen,_pix,False,line,2)
        
        for line in self.lines:
            line[0][:,1] += self.offset

class BlurVisual(Visual):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with open('blurred_pumpkin','rb') as f:
            self.images = pickle.load(f)
        w,h,c = np.shape(self.images[0])
        self.surfs = [pygame.Surface((w,h)) for i in self.images]
        [surfarray.blit_array(surf,image) for surf,image in zip(self.surfs,self.images)]
        s  = self.surfs[0]
        self.screen.blit(s,(0,0))
        self.L = len(self.images)
        self.maximum = 0
        
    def iterate(self,t):
        vol = self.stream.volume()
        self.maximum = max(vol,self.maximum)
        print(self.maximum,vol)
        idx = min(0,max(self.L,int(vol/1000)))
        s  = self.surfs[idx]
        self.screen.blit(s,(0,0))
        self.screen.fill((0,0,0))

class GravityVisual(Visual):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.N     = 4
        self.radii = np.random.randint(20,50,size=self.N)
        self.pos   = [(self.radii[0],self.radii[0]),
                      (self.width-self.radii[1],self.radii[1]),
                      (self.radii[2],self.height-self.radii[2]),
                      (self.width-self.radii[3],self.height-self.radii[3])]
        self.rate  = np.zeros((self.N,2))
        self.color = np.random.randint(0,256,size=(self.N,3))
        self.cfreq = np.random.random(size=(self.N,3))
        self.bm    = BallManager(self.N,self.pos,self.rate,self.radii,
                                 self.color,self.cfreq,self.width,self.height)
        self.black = np.zeros((width,height,3))

    def iterate(self,t):
        surfarray.blit_array(self.screen, self.black)
        self.bm.draw_all(self.screen)
        self.bm.update(t)

def enable_fullscreen():
    pygame.display.quit()
    pygame.display.init()
    return pygame.display.set_mode((0,0),pygame.FULLSCREEN)

def disable_fullscreen(size):
    pygame.display.quit()
    pygame.display.init()
    return pygame.display.set_mode(size, pygame.RESIZABLE)

def reset_visuals(screen,stream,visuals):
    [visual.__init__(screen,stream) for visual in visuals]

def run_visuals(stream_manager,size):
    t = 0
    pygame.init()
    screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)
    visuals = [cls(screen,stream_manager) for cls in
               (BasicVisual,JoyDivisionVisual,BlurVisual,GravityVisual)]
    
    transition = 0
    visual = 0
    running = True
    while running:
        print(t)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type is pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        if not transition:
            visuals[visual].iterate(t)
            
            if t > 10000:
                #visual = np.random.randint(0,3)
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
    print("shutting it down")

size = width, height = 1200, 500

FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 44100
CHUNK    = 512

sm = StreamManager(FORMAT,CHANNELS,RATE,CHUNK)
sm.start()

try:
    run_visuals(sm,size)
except KeyboardInterrupt:
    print('User interruption, shutting down stream...')
    pass

sm.stop()
