import numpy as np 

X = np.arange(100).reshape(4,25)

np.savetxt('center.txt', X, delimiter = ',')
