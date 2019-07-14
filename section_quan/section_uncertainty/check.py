import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.stats import expon

a = []
b = []
d1 = expon(scale=1.0/60.0)
d2 = expon(scale=60.0)
bins = np.linspace(-5.0, 5.0, 50)

print("Start drawing")
start = time.time()
for i in range(10000):
    if i%100==0: print(i)
    #a.append(expon(scale=1/60.0).rvs())
    #b.append(expon(scale=60.0).rvs())
    a.append(d1.rvs())
    b.append(d2.rvs())
stop = time.time()
print("finished drawing")
print("Time:", stop - start)


#plt.hist([a, b], 100, density=1, alpha=0.6, label=['60.0', '1/60.0'])
plt.hist(a, 100, density=1, alpha=0.6, label='1/60.0')
#plt.hist(b, 100, density=1, alpha=0.6, label='1/60.0')
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
