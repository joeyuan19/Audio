import time
import numpy as np

a = list(range(1000))
b = list(range(2000,3000))

arr1 = np.array([a,b]).T
arr2 = np.array([(i,j) for i,j in zip(a,b)])
arr3 = np.array([[a],[b]])

print(arr1)
print(arr2)
print(arr3)
print(arr1==arr2,arr2==arr3)

t0 = time.time()
arr = np.array([a,b]).T
print(time.time() - t0)

t0 = time.time()
arr = np.array([(i,j) for i,j in zip(a,b)])
print(time.time() - t0)

