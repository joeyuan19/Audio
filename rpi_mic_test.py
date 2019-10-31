import pyaudio
import numpy as np
import time
import wave


# open stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

CHUNK = 2048 # RATE / number of updates per second

RECORD_SECONDS = 20


# use a Blackman window
window = np.blackman(CHUNK)

x = 0

m = 0

def soundPlot(stream,m):
    t1=time.time()
    data = stream.read(CHUNK, exception_on_overflow=False)
    waveData = wave.struct.unpack("%dh"%(CHUNK), data)
    npArrayData = np.array(waveData)
    a = np.average(np.abs(npArrayData))
    m = max(m,a)
    print(m,a)
    indata = npArrayData*window
    print("took %.02f ms"%((time.time()-t1)*1000))
    return m

if __name__=="__main__":
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNK)

    m = 0
    for i in range(0, RATE // CHUNK * RECORD_SECONDS):
        m = soundPlot(stream,m)

    stream.stop_stream()
    stream.close()
    p.terminate()
