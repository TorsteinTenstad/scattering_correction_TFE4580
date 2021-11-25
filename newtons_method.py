import numpy as np
from matplotlib import pyplot as plt
from read_data import get_reflectance
import mie
from tqdm import tqdm

def newtons_method(f_x, gradient, x_0, epsilon = 1e-10, max_iter = 10, minimum_defined_value=None):
    x_n = np.array(x_0)
    for n in range(max_iter):
        f = f_x(x_n)
        error = abs(f)
        if error < epsilon:
            return x_n, error
        df = np.array(gradient(x_n))
        step = -df*(f/(np.sum(df))**2)
        x_n = x_n + step
        if minimum_defined_value and x_n < minimum_defined_value:
            x_n = minimum_defined_value
    print('Exceeded maximum iterations. No solution found.')
    return x_n, error


def approximate_gradient(f_x, x, step=4):
    f1 = f_x(x+step)
    f0 = f_x(x)
    d = (f1-f0)/step
    print('f0: ', f0, '\tf1: ', f1, '\tderivative:', d)
    return d


def newtons_method_using_approximate_gradient(f_x, x_0, epsilon = 1e-10, max_iter = 100):
    gradient = lambda x : approximate_gradient(f_x, x)
    return newtons_method(f_x, gradient, x_0, epsilon, max_iter)


def alternating_fit(f_x, X, x_dependent_parameter, indepedent_parameters, step=1e-15, epsilon = 1e-3, max_iter = 100):
    for n in range(0,max_iter):
        for i, x in enumerate(X):
            f_a = lambda a : f_x(x, np.array([a]), indepedent_parameters)
            x_dependent_parameter[i], error = newtons_method_using_approximate_gradient(f_a, np.array([x_dependent_parameter[i]]), epsilon)
        f_theta = lambda indepedent_parameters : np.sqrt(np.sum(f_x(X, x_dependent_parameter, indepedent_parameters)**2))
        indepedent_parameters, error = newtons_method_using_approximate_gradient(f_theta, indepedent_parameters, epsilon)
        print(error)
        if error < epsilon:
            return x_dependent_parameter, indepedent_parameters






if __name__ == "__main__":
    from reflectance_model import D_gamma, gamma_model, musr_model, gamma
    centers, rois = get_reflectance(3, 3)

    fig, axs = plt.subplots(2)
    ax2, ax = axs
    for c, roi in enumerate(rois.values()):
        for spectrum in roi[0:20]:
            print('.', end='')
            spectrum = 0.8*spectrum

            r = 1e-5
            A = 0.17
            ns = 1.4
            nb = 1
            vol = 4*np.pi*r**3/3
            Ns = 0.56/vol

            mua = 20*np.ones(len(centers))
            musr_ = np.empty(len(centers))
            gamma_ = np.empty(len(centers))
            Qs_ = np.empty(len(centers))

            mu_s_prime = mie.get_mu_s_prime_data(r, centers, ns, nb, Ns)

            for i, lambda_ in enumerate(centers):
                f = lambda mua : gamma(A, mu_s_prime[i], mua) - spectrum[i]
                D_f = lambda mua : D_gamma(A, mu_s_prime[i], mua)
                mua[i] = newtons_method(f, D_f, mua[i], epsilon=1e-4, minimum_defined_value=0.001)[0]
                gamma_[i] = gamma(A, mu_s_prime[i], mua[i])
            ax.plot(centers, mua, label='mua', color='C'+str(c))
            ax2.plot(centers, spectrum, label='reflectance', color='C'+str(c))
    plt.show()