import numpy as np
from numpy.core.function_base import linspace
from numpy.matrixlib import defmatrix
from scipy import special
from matplotlib import pyplot as plt
from read_data import get_reflectance, read_data
from newtons_method import newtons_method
import mie
from a import a

np.seterr(all = "raise")

def gamma(A, musr, mua):
    gamma_ = A*musr/((1+A)*mua+A*musr+(1/3+A)*np.sqrt(3*musr*mua+3*mua**2))
    return gamma_

def D_gamma(A, musr, mua):
    gamma_ = gamma(A, musr, mua)
    D_gamma_ = -gamma_**2/(A*musr)*((1+A)+((A+1/3)*(3*musr+6*mua))/(2*np.sqrt(3*musr*mua+3*mua**2)))
    return D_gamma_

def musr(g, Ns, Qs, r):
    return (1-g)*Ns*Qs*np.pi*r**2

def g(Qs, x, als, bls):
    l = np.arange(len(als))[1:]
    als_shifted = als[1:]
    bls_shifted = bls[1:]
    als = als[:-1]
    bls = bls[:-1]
    summand = (l*(l+2)/(l+1))*np.real(als*np.conj(als_shifted)+bls*np.conj(bls_shifted))+((2*l+1)/(l*(l+1)))*np.real(als*np.conj(bls))
    return (4/(Qs*(x**2)))*np.sum(summand)

def Qs(x, als, bls):
    l = np.arange(len(als)) + 1
    return (2/(x**2))*(np.sum((2*l+1)*(np.absolute(als)**2+np.absolute(bls)**2)))

def als(x, y, nr):
    return ab_helper(x, y, 1, nr)

def bls(x, y, nr):
    return ab_helper(x, y, nr, 1)

def ab_helper(x, y, nr_b, nr_a):
    infinity_aproximation = 1000
    coefficients = np.zeros(infinity_aproximation, dtype=np.complex128)
    for i in range(infinity_aproximation):
        l = i+1
        numerator   = nr_b*D_phi(l, y)* phi(l, x)-nr_a*phi(l,y)*D_phi(l,x)
        denominator = nr_b*D_phi(l, y)*zeta(l, x)-nr_a*phi(l,y)*D_zeta(l,x)
        if np.isnan(numerator) or np.isnan(denominator):
            print('OI', i)
            return coefficients
        coefficients[i] = numerator/denominator
        if coefficients[i] < 1e-12:
            print('MATE', i)
            return coefficients
    return coefficients

def x(r, nb, lambda_):
    print(2*r*np.pi*nb/lambda_)
    return 2*r*np.pi*nb/lambda_

def y(r, ns, lambda_):
    return 2*r*np.pi*ns/lambda_

def phi(l, z):
    return np.sqrt(np.pi*z/2)*J(l+1/2, z)

def xi(l, z):
    return -np.sqrt(np.pi*z/2)*Y(l+1/2, z)

def zeta(l, z):
    return phi(l, z) + 1j*xi(l, z)

def D_phi(l, z):
    return ((1/2-l)/z)*phi(l, z) + phi(l-1, z)
    #return phi(l, z)/(2*z) + phi(l-1, z)/(2) + phi(l+1, z)/(2)

def D_zeta(l, z):
    return ((1/2-l)/z)*zeta(l, z) + zeta(l-1, z)
    #return zeta(l, z)/(2*z) + zeta(l-1, z)/(2) + zeta(l+1, z)/(2)

def J(v, z):
    #print('J:', v, z, special.jv(v, z))
    return special.jv(v, z)

def Y(n, z):
    #print('Y:', n, z, special.yv(n, z))
    return special.yv(n, z)


def gamma_model(A, r, ns, nb, Ns, mua, lambdas):
    n = len(lambdas)
    xs = np.empty(n)
    ys = np.empty(n)
    Qss = np.empty(n)
    gs = np.empty(n)
    musrs = np.empty(n)
    gammas = np.empty(n)
    
    for i, lambda_ in enumerate(lambdas):
        xs[i] = x(r, nb, lambda_)
        ys[i] = y(r, ns, lambda_)
        nr = ns/nb
        als_ = als(xs[i], ys[i], nr)
        bls_ = bls(xs[i], ys[i], nr)
        Qss[i] = Qs(xs[i], als_, bls_)
        gs[i] = g(Qss[i], xs[i], als_, bls_)
        musrs[i] = musr(gs[i], Ns, Qss[i], r)
        gammas[i] = gamma(A, musrs[i], mua[i])
    return xs, ys, Qss, gs, musrs, gammas


def musr_model(r, ns, nb, Ns, lambdas):
    n = len(lambdas)
    xs = np.empty(n)
    ys = np.empty(n)
    Qss = np.empty(n)
    gs = np.empty(n)
    musrs = np.empty(n)

    for i, lambda_ in enumerate(lambdas):
        xs[i] = x(r, nb, lambda_)
        ys[i] = y(r, ns, lambda_)
        nr = ns/nb
        als_ = als(xs[i], ys[i], nr)
        bls_ = bls(xs[i], ys[i], nr)
        Qss[i] = Qs(xs[i], als_, bls_)
        gs[i] = g(Qss[i], xs[i], als_, bls_)
        musrs[i] = musr(gs[i], Ns, Qss[i], r)
    return Qss, musrs


def gamma_model_x(xs, nr):
    n = len(xs)
    ys = nr*xs
    Qss = np.empty(n)
    gs = np.empty(n)
    
    for i, x in enumerate(xs):
        als_ = als(xs[i], ys[i], nr)
        bls_ = bls(xs[i], ys[i], nr)
        Qss[i] = Qs(xs[i], als_, bls_)
        gs[i] = g(Qss[i], xs[i], als_, bls_)
    return xs, ys, Qss, gs


def fit_gamma_model(lambdas, spectrum, mua_0=None):
    ns = 1.44
    nb = 1
    A = a(ns, nb)

    mua = mua_0 if mua_0 else 20*np.ones(len(lambdas))
    gamma_ = np.empty(len(lambdas))

    rs = [0.8, 0.7, 0.51, 0.285, 0.125, 0.085, 0.07]
    pr = [0.2, 4.8, 25, 57, 8, 3.5, 1.5]
    mu_s_prime = np.zeros(len(lambdas))
    for r, p in zip(rs, pr):
        vol = 4*np.pi*r**3/3
        Ns = 0.59/vol
        mu_s_prime += p*mie.get_mu_s_prime_data(r, lambdas, ns, nb, Ns)

    for i, lambda_ in enumerate(lambdas):
        f = lambda mua : gamma(A, mu_s_prime[i], mua) - spectrum[i]
        D_f = lambda mua : D_gamma(A, mu_s_prime[i], mua)
        mua[i] = newtons_method(f, D_f, mua[i], epsilon=1e-4, minimum_defined_value=0.001)[0]
        gamma_[i] = gamma(A, mu_s_prime[i], mua[i])
    return mua, mu_s_prime


if __name__ == "__main__":
    mua = np.linspace(0.02, 2000, 1000)
    musr_ = 1
    gamma_ = gamma(0.17, musr_, mua)
    fig, ax = plt.subplots()
    ax.plot(mua, gamma_)
    ax.set_yscale('log')
    plt.show()

    exit()
    lambdas, rois, concentrations = get_reflectance(100, 100)
    spectrum = np.average(rois[0], axis=0)
    mua, mu_s_prime = fit_gamma_model(lambdas, spectrum)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(lambdas, spectrum, label='reflectance', color='C0')
    ax2.plot(lambdas, mua, label='mua', color='C1')
    ax2.plot(lambdas, mu_s_prime, label='musr', color='C2')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()


    exit()
    r = 1e-5
    vol_s = 4*np.pi*r**3/3
    Ns = 0.56/vol_s
    xs, ys, Qss, gs, musrs, gammas = gamma_model(0.17, r, 1.5, 1, Ns, mua, lambdas)
    X = 10**np.linspace(0, 2.2, 100)
    xs, ys, Qss, gs = gamma_model_x(X, 1.18)
    plt.plot(np.linspace(0, 2.2, 100), np.log10(Qss))
    plt.show()

