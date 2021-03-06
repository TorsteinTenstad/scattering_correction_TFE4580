from re import L
import numpy as np
from scipy import special
from matplotlib import pyplot as plt
import pickle
import os
from tqdm import tqdm
from butter_filter import butter_lowpass_filter
from read_data import read_data

def phi(l, z):
    return np.sqrt(np.pi*z/2)*J(l+1/2, z)

def xi(l, z):
    return -np.sqrt(np.pi*z/2)*Y(l+1/2, z)

def zeta(l, z):
    return phi(l, z) + 1j*xi(l, z)

def D_phi(l, z):
    return ((1/2-l)/z)*phi(l, z) + phi(l-1, z)

def D_zeta(l, z):
    return ((1/2-l)/z)*zeta(l, z) + zeta(l-1, z)

def J(v, z):
    return special.jv(v, z)

def Y(n, z):
    return special.yv(n, z)

def Qs_and_g(x, n_rel):  
    y = n_rel*x

    err = 1e-8

    Qs = 0
    gQs = 0
    for n in range(1, 100000):
        Snx = phi(n, x)
        Sny = phi(n, y)
        Zetax = zeta(n, x)
        Snx_prime = D_phi(n, x)
        Sny_prime = D_phi(n, y)
        Zetax_prime = D_zeta(n, x)

        an_num = Sny_prime*Snx-n_rel*Sny*Snx_prime
        an_den = Sny_prime*Zetax-n_rel*Sny*Zetax_prime
        an = an_num/an_den

        bn_num = n_rel*Sny_prime*Snx-Sny*Snx_prime
        bn_den = n_rel*Sny_prime*Zetax-Sny*Zetax_prime
        bn = bn_num/bn_den

        Qs1 = (2*n+1)*(np.abs(an)**2+np.abs(bn)**2)
        Qs = Qs + Qs1
        if n > 1:
            gQs1 = (n-1)*(n+1)/n*np.real(an_1*np.conj(an)+bn_1*np.conj(bn))+(2*n-1)/((n-1)*n)*np.real(an_1*np.conj(bn_1))
            gQs = gQs + gQs1
        
        an_1 = an
        bn_1 = bn

        if np.abs(Qs1)<(err*Qs) and np.abs(gQs1)<(err*gQs):
            break
    Qs = (2/x**2)*Qs
    gQs = (4/x**2)*gQs
    g = gQs/Qs

    return Qs, g


def create_Qs_and_g_path(x_start, x_end, n_rel, n, smooth):
    smooth_identifier = 'smooth_' if smooth else ''
    path = 'Qs_and_g/' + smooth_identifier + '%.2f_%.2f_%.4f_%d.pickle' % (x_start, x_end, n_rel, n)
    return path


def create_Qs_and_g_graph(x_start, x_end, n_rel, n):
    path = create_Qs_and_g_path(x_start, x_end, n_rel, n, False)
    xs = np.linspace(x_start, x_end, n)
    Qss = np.empty(n)
    gs = np.empty(n)
    for i, x in tqdm(enumerate(xs)):
        Qss[i], gs[i] = Qs_and_g(x, n_rel)
    data = (xs, Qss, gs)
    with open(path, 'wb') as output_file:
        pickle.dump(data, output_file)
    return data


def get_Qs_and_g_graph(x_start, x_end, n_rel, n):
    path = create_Qs_and_g_path(x_start, x_end, n_rel, n, False)
    if os.path.exists(path):
        with open(path, 'rb') as input_file:
            data = pickle.load(input_file)
            xs, Qss, gs = data
        return xs, Qss, gs
    else:
        return create_Qs_and_g_graph(x_start, x_end, n_rel, n)


def get_smooth_Qs_and_g_graph(x_start, x_end, n_rel, n):
    xs, Qss, gs = get_Qs_and_g_graph(x_start, x_end, n_rel, n)
    Qss = butter_lowpass_filter(Qss, 50, n)
    gs = butter_lowpass_filter(gs, 50, n)
    return xs, Qss, gs



def x_and_n_rel(r, lambdas, n_s, n_b):
    k = 2*np.pi*n_b/lambdas
    x = k*r
    n_rel = n_s/n_b
    return x, n_rel


def mu_s_prime(Qs, g, r, N_s):
    sigma_s = Qs*np.pi*r**2
    mu_s = N_s*sigma_s
    mu_s_prime = mu_s*(1-g)
    return mu_s_prime


def get_mu_s_prime_data(r, lambdas, ns, nb, Ns):
    x_eval, n_rel = x_and_n_rel(r, lambdas, ns, nb)

    #limits = np.genfromtxt('limits.txt')
    #limits[0] = max(limits[0], np.max(x_eval))
    #limits[1] = min(limits[1], np.min(x_eval))
    #np.savetxt('limits.txt', limits)

    x_min = 1e+05
    x_max = 6e+06
    xs, Qss, gs = get_smooth_Qs_and_g_graph(100, 6000, n_rel, n=1000)
    Qs = np.interp(x_eval, xs, Qss)
    g = np.interp(x_eval, xs, gs)
    mu_s_prime_ = mu_s_prime(Qs, g, r, Ns)
    return mu_s_prime_




if __name__ == "__main__":
    x_min = 1.748331551456249435e+05
    x_max = 5.278096071058874950e+06
    x_min = 100
    x_max = 6000
    xs, Qss, gs = get_Qs_and_g_graph(x_min, x_max, 1.44, n=1000)
    plt.plot(xs, Qss)
    plt.show()
    xs, Qss, gs = get_smooth_Qs_and_g_graph(x_min, x_max, 1.44, n=1000)
    plt.plot(xs, Qss)
    plt.show()

    exit()
    lambdas, _, _ = read_data()
    r = 0.8
    vol = 4*np.pi*r**3/3
    Ns = 0.59/vol
    musr = get_mu_s_prime_data(r, lambdas, 1.44, 1, Ns)
    plt.plot(lambdas, musr)
    plt.show()




