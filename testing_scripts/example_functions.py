import numpy as np
from matplotlib import pyplot as plt

def hyperbel(theta):
    x, y = theta
    return x**2+y**2


def hyperbel_gradient(theta):
    x, y = theta
    ddx = 2*x
    ddy = 2*y
    return ddx, ddy


def waves(theta):
    x, y = theta
    return np.sin(x) + np.sin(y)


def waves_gradient(theta):
    x, y = theta
    return np.cos(x), np.cos(y)


def h(theta):
    x = theta[0]
    return np.sin(x)*x**2*np.exp(-x)+0.1


def g(x, a_x, theta):
    return 3+x-theta*a_x

def g_target(x):
    a_x = np.log10(x**2+2)
    theta = 5
    return g(x, a_x, theta)

def f(x, a_x, theta):
    return g(x, a_x, theta)-g_target(x)


if __name__ == "__main__":
    x = np.linspace(0, 10, 25)
    plt.plot(x, g_target(x))
    plt.show()
