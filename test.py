from tabulate import tabulate
k = []
for i in range(100):
    k.append(i)
h = []
for i in range(0, len(k), 10):  
    h.append(k[i:i + 10])

print(h)