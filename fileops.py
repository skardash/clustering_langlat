import numpy as np
f = open('data', 'r')
f.readline()
dt = []
for line in f:
    splt = line.split()
    dt.append([float(splt[1]), float(splt[2])])
print np.array(dt)