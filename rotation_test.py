import matplotlib.pyplot as plt
import numpy as np


def rotation(v,angle):
    c,s = np.cos(angle),np.sin(angle)
    return np.dot(v,np.array([[c,-s],[s,c]]))

v = np.array((1,10))
plt.plot((0,v[0]),(0,v[1]))

a = (0,0)
b = (5,1)

r = np.sqrt(b[0]**2 + b[1]**2)
p = np.arctan2(b[1],b[0])

pts = np.array([a,b]).T
plt.plot(pts[0],pts[1])

_v = rotation(rotation(v,p)*np.array((-1,1)),-p)
plt.plot((0,_v[0]),(0,_v[1]))

plt.xlim((-6,6))
plt.ylim((-6,6))

plt.show()
