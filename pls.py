from sklearn.cross_decomposition import PLSRegression
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pickle


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


sets = {}
with open('results.pickle', 'rb') as input_file:
    sets = pickle.load(input_file)

lambdas = sets['lambdas']
concentrations = sets['concentrations']
train_set = sets['train_set']
test_set = sets['test_set']
EMSC_corrected_train = sets['EMSC_corrected_train']
EMSC_corrected_test = sets['EMSC_corrected_test']
optically_corrected_train = sets['optically_corrected_train']
optically_corrected_test = sets['optically_corrected_test']
ref_0 = sets['ref_0']
ref_100 = sets['ref_100']


nrows=3
fig_spectra, axs_spectra = plt.subplots(nrows=nrows, sharex='all')
fig_pls, axs_pls = plt.subplots(nrows=nrows, sharex='all')

pls_evaluation(lambdas*1e9, concentrations, train_set, test_set, axs_spectra[0], axs_pls[0])
pls_evaluation(lambdas*1e9, concentrations, EMSC_corrected_train, EMSC_corrected_train, axs_spectra[1], axs_pls[1])
pls_evaluation(lambdas*1e9, concentrations, optically_corrected_train, optically_corrected_train, axs_spectra[2], axs_pls[2])

titles = ['Uncorrected extinction', 'Extinction corrected by EMSC', 'Absorption estimated with optical model']
titles = ['Raw data', 'Data corrected with method 1', 'Data corrected with method 2']

for i, title in enumerate(titles):
    axs_spectra[i].set_yscale('log')
    axs_pls[i].set_ylabel('Predicted concentration [%]')
    axs_spectra[i].text(0.04, 0.9, title, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round", ec=(0.8, 0.8, 0.8),fc=(0.9, 0.9, 0.9),), size='x-large', transform=axs_spectra[i].transAxes)
    axs_pls[i].text(0.04, 0.9, f'Using {title[0].lower()}{title[1:]}', horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round", ec=(0.8, 0.8, 0.8),fc=(0.9, 0.9, 0.9),), size='x-large', transform=axs_pls[i].transAxes)
axs_spectra[-1].set_xlabel('Wavelength [nm]')
axs_pls[-1].set_xlabel('True concentration [%]')

lines = [Line2D([0], [0], color='C%d'%i, lw=4) for i, c in enumerate(concentrations)]
labels = ['%.1f' % c for c in concentrations]
#fig_pls.legend(lines, labels, loc='center left', title='Concentrations')
#fig_spectra.legend(lines, labels, loc='center left', title='Concentrations')
plt.show()

