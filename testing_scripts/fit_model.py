from scipy import special
from newtons_method import newtons_method, newtons_method_using_approximate_gradient
import reflectance_model
import read_data
import numpy as np
from matplotlib import pyplot as plt


def model(target, parameters, lambdas):
    r = parameters[0]
    mua = parameters[1:]
    vol_s = 4*np.pi*r**3/3
    Ns = 0.56/vol_s
    f = target - reflectance_model.gamma_model(0.17, r, 1.5, 1, Ns, mua, lambdas)
    return f


if __name__ == "__main__":
    centers, rois = read_data.read_data()
    spectrum = np.average(rois[0.0], axis=(0,1))
    f = lambda parameters : model(spectrum, parameters, centers)
    starting_parameters = np.array([1e-4] + len(spectrum)*[200])
    parameter_approximation = newtons_method_using_approximate_gradient(f, starting_parameters)

    plt.plot(centers, spectrum)
    plt.show()