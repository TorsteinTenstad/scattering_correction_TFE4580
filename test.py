import numpy as np
from matplotlib import pyplot as plt
from read_data import read_data


def n_sio2(lambdas):
    microns = lambdas*1e6
    num_1 = 0.6961663*microns**2
    den_1 = microns**2-0.06884043**2
    num_2 = 0.4079426*microns**2
    den_2 = microns**2-0.1162414**2
    num_3 = 0.8974794*microns**2
    den_3 = microns**2-9.896161**2
    n = np.sqrt(num_1/den_1+num_2/den_2+num_3/den_3+1)
    return n


if __name__ == "__main__":
    centers, _, _ = read_data()
    l = 1e-9*np.linspace(300, 4000, 500)
    l_ = np.linspace(centers[0], centers[-1], 500)

    fig, ax = plt.subplots()
    

    ax.plot(1e9*l, n_sio2(l))
    ax.plot(1e9*l_, n_sio2(l_), label='Wavelength range of the hyperspectral images')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Refractive index of SiO2')
    plt.legend()
    plt.show()