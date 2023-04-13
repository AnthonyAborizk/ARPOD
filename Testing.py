# Testing bc I'm a newb
import numpy as np
a = 1
b = 3
dones = np.zeros(2,)
dones[a + b > 1] = 1
dones = True
print(dones)