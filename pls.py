from numpy import nan
from sklearn.cross_decomposition import PLSRegression
from EMSC import binary_EMSC
from read_data import get_reflectance, get_transmission
from read_data import get_absorbance_data
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle
from matplotlib.lines import Line2D

from reflectance_model import fit_gamma_model

def plot_spectra(ax, lambdas, spectra):
    for c, spectra_c in enumerate(spectra):
        for spectrum in spectra_c:
            ax.plot(lambdas, spectrum, color='C%d'%c)

def pls_evaluation(lambdas, concentrations, train_set, test_set, ax_spectra=None, ax_pred=None, n_components=2):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for c in range(len(train_set)):
        for spectrum in train_set[c]:
            train_x.append(spectrum)
            train_y.append(concentrations[c])
        for spectrum in test_set[c]:
            test_x.append(spectrum)
            test_y.append(concentrations[c])
            if ax_spectra:
                ax_spectra.plot(lambdas, spectrum, color='C%d'%c, linewidth=0.2)

    pls2 = PLSRegression(n_components)
    pls2.fit(train_x, train_y)

    pred_y = pls2.predict(test_x)
    mse = 0
    n = len(test_x)
    for i in range(n):
        mse += abs(pred_y[i] - test_y[i])**2/n
        pred_y[i][0], test_y[i]
    if ax_pred:
        l = [np.min(test_y), np.max(test_y)]
        ax_pred.plot(l, l, color='black')
        for i, c in enumerate(test_y):
            ax_pred.plot(c, pred_y[i], '.', color='C%d'%np.where(concentrations==c)[0])
        ax_pred.text(100, 10, 'MSE = %.2f^2' % np.sqrt(mse), horizontalalignment='right', verticalalignment='bottom', bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),), size='x-large')
    return mse

def apply_func_on_sets(train_set, test_set, func):
    output = [None, None]
    for t, set in enumerate([train_set, test_set]):
        output[t] = [None]*len(set)
        for c, spectra_c in enumerate(set):
            output[t][c] = [None]*len(spectra_c)
            for i, spectrum, in enumerate(spectra_c):
                output[t][c][i] = func(spectrum)
    return output[0], output[1]

lambdas, spectra, concentrations = get_transmission(12, 12)
absorbance_sio2, absorbance_caco3 = get_absorbance_data(lambdas)

train_set = []
test_set = []

for c, spectra_c in enumerate(spectra):
        half = int(len(spectra_c)/2)
        shuffle(spectra_c)
        train_set.append([])
        test_set.append([])
        for spectrum in spectra_c[0:half]:
            train_set[c].append(spectrum)
        for spectrum in spectra_c[half:]:
            test_set[c].append(spectrum)

ref_0 = np.average(train_set[0], axis=0)
ref_100 = np.average(train_set[-1], axis=0)

'''
EMSC_train_h, EMSC_test_h = apply_func_on_sets(train_set, test_set, lambda s : binary_EMSC(lambdas, ref_0, ref_100, s)[1])

n = 0
mse = 0
for c, h_c in enumerate(EMSC_test_h):
    for h in h_c:
        mse += abs(concentrations[c] - 100*(0.5-h))**2
        n+=1
        print(concentrations[c], 100*(0.5-h))
mse = mse/n
sqrt_mse = np.sqrt(mse)
print('sqrt_mse:', sqrt_mse)
'''

EMSC_corrected_train, EMSC_corrected_test = apply_func_on_sets(train_set, test_set, lambda s : binary_EMSC(lambdas, ref_0, ref_100, s)[0])

optically_corrected_train, optically_corrected_test = apply_func_on_sets(train_set, test_set, lambda s : fit_gamma_model(lambdas, 1-s)[0])

nrows=3
fig_spectra, axs_spectra = plt.subplots(nrows=nrows, sharex='all')
fig_spectra.subplots_adjust(top=1, bottom=0.035, left=0.16, right=0.998, hspace=0, wspace=0)
fig_pls, axs_pls = plt.subplots(nrows=nrows, sharex='all')
fig_pls.subplots_adjust(top=1, bottom=0.035, left=0.05, right=0.998, hspace=0, wspace=0)
print(pls_evaluation(lambdas*1e9, concentrations, train_set, test_set, axs_spectra[0], axs_pls[0]))
print(pls_evaluation(lambdas*1e9, concentrations, EMSC_corrected_train, EMSC_corrected_train, axs_spectra[1], axs_pls[1]))
#print(pls_evaluation(lambdas, concentrations, log_EMSC_corrected_train, log_EMSC_corrected_test, axs[2][0], axs[2][1]))
print(pls_evaluation(lambdas*1e9, concentrations, optically_corrected_train, optically_corrected_train, axs_spectra[2], axs_pls[2]))

for i, title in enumerate(['Uncorrected extinction', 'Extinction corrected by EMSC', 'Absorption estimated with optical model']):
    axs_spectra[i].set_yscale('log')
    axs_pls[i].set_ylabel('Predicted concentration [%]')
    axs_spectra[i].text(0.04, 0.9, title, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round", ec=(0.8, 0.8, 0.8),fc=(0.9, 0.9, 0.9),), size='x-large', transform=axs_spectra[i].transAxes)
    axs_pls[i].text(0.04, 0.9, f'Using {title[0].lower()}{title[1:]}', horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round", ec=(0.8, 0.8, 0.8),fc=(0.9, 0.9, 0.9),), size='x-large', transform=axs_pls[i].transAxes)
axs_spectra[-1].set_xlabel('Wavelength [nm]')
axs_pls[-1].set_xlabel('True concentration [%]')


lines = [Line2D([0], [0], color='C%d'%i, lw=4) for i, c in enumerate(concentrations)]
labels = ['%.1f' % c for c in concentrations]
#fig_pls.legend(lines, labels, loc='center left', title='Concentrations')
fig_spectra.legend(lines, labels, loc='center left', title='Concentrations')
plt.show()