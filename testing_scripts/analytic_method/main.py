from numpy.core.numeric import zeros_like
from ReflE2L2 import ReflE2L2_analytic
from skindata import skindata
import numpy as np
from matplotlib import pyplot as plt


f_min = 400
f_max = 850


f = np.arange(f_min, f_max, 1)
x = np.zeros(f_max-f_min)

for i, lambda_ in enumerate(f):
    ip = skindata(lambda_)
    x[i] = ReflE2L2_analytic(ip)

plt.plot(f,x)
plt.show()