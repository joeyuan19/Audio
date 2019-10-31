import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter
import pickle
from itertools import repeat

from multiprocessing.dummy import Pool as ThreadPool

filepath = '/Users/joe_yuan/Desktop/Desktop/Projects/Audio/pumpkin.jpg'
im = plt.imread(filepath)
fig,(ax1,ax2) = plt.subplots(1,2)

ax1.imshow(im)

images = []


def blur_image(im,sigma):
    print(sigma)
    _im = im.copy()
    for i in range(3):
        _im[:,:,i] = gaussian_filter(_im[:,:,i],sigma=sigma)
    return np.transpose(_im,(1,0,2))

#for s in range(100):
#    print(s)
#    test = im.copy()
#    for i in range(3):
#        test[:,:,i] = gaussian_filter(test[:,:,i],sigma=s)
#    images.append(test.copy().T)

with ThreadPool(4) as pool:
    images = pool.starmap(blur_image,zip(repeat(im),list(range(100))))

plt.imshow(images[0])
plt.show()

import sys
sys.exit()

with open('blurred_pumpkin','wb') as f:
    pickle.dump(images,f)
