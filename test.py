'''
from tabulate import tabulate
k = []
for i in range(100):
    k.append(i)
h = []
for i in range(0, len(k), 10):  
    h.append(k[i:i + 10])

print(tabulate(h))


# This is just a test

'''
import numpy as np
from tabulate import tabulate

a = np.array([[0,0,0],[1,0,0],[1,0,0]])
print(tabulate(a))

a = a.reshape(1,9)
print(tabulate(a))
a = a.reshape(3,3)

print(tabulate(a))

#b = np.argmin(np.norm(a, axis=0))

#print(b)
