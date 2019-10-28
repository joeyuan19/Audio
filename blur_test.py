import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter
import pickle

filepath = '/Users/joe_yuan/Desktop/Desktop/Projects/Audio/pumpkin.jpg'
im = plt.imread(filepath)
fig,(ax1,ax2) = plt.subplots(1,2)

ax1.imshow(im)

images = []

for s in range(100):
    print(s)
    test = im.copy()
    for i in range(3):
        test[:,:,i] = gaussian_filter(test[:,:,i],sigma=s)
    images.append(test.copy())

with open('blurred_pumpkin','wb') as f:
    pickle.dump(images,f)
