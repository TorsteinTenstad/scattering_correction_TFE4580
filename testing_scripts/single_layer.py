import numpy as np
from matplotlib import pyplot as plt
import read_data


def reflectance(mua, musr):
    A = 0.17
    mutr = mua + musr
    delta = 1/(3*mua*mutr)
    D = 1/(3*mutr)
    return (musr*A*delta**2)/((delta/(3*D)+1)+1*(D+delta*A))


lambda_min = 400
lambda_max = 850
lambda_ = np.arange(lambda_min, lambda_max, 1)

musr = 1
mua = 1

plt.plot(lambda_, reflectance(mua, musr))
plt.show()
