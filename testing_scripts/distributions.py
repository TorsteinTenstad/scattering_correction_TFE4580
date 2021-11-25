import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


def gaussian(mu=None, cov=None, r=10):
    if mu is None:
        mu = np.zeros(2)
    if cov is None:
        cov = np.ones(2)

    rv = multivariate_normal(mu, cov)
    x, y = np.mgrid[mu[0]-r:mu[0]+r+1, mu[1]-r:mu[1]+r+1]
    xy = np.dstack((x, y))
    p = rv.pdf(xy)
    return x, y, p

def zero_mean_uncorrelated_gaussian(sigma=None):
    if sigma is None:
        sigma=1
    return gaussian(cov=sigma*np.eye(2), r=int(3*np.sqrt(sigma)))


if __name__ == "__main__":
    x, y, p = zero_mean_uncorrelated_gaussian(0.2)
    print(p)
    plt.contourf(x, y, p)
    plt.show()