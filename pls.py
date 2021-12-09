from numpy.core.numeric import full
from numpy.lib import average
from sklearn.cross_decomposition import PLSRegression
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import cm
import pickle
from matplotlib import rc
#rc('text', usetex=True) 


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

methods = ['Raw apparent absorption', 'Absorption after EMSC correction', 'Absorption coefficient estimated by optical model']
my_dpi = 140
default_figsize = (1920/my_dpi, 2160/my_dpi)


cmap = cm.get_cmap('plasma')
c_colors = [cmap(i/len(concentrations)) for i in range(len(concentrations))]
set_colors = ['C5', 'C0', 'C2']

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
                ax_spectra.plot(lambdas, spectrum, color=cmap(c/len(concentrations)), linewidth=0.2)

    pls2 = PLSRegression(n_components)
    pls2.fit(train_x, train_y)

    pred_y = pls2.predict(test_x)
    mse = 0
    n = len(test_x)
    guesses = {i: [] for i in range(len(concentrations))}
    for estimate, true in zip(pred_y, test_y):
        mse += abs(estimate - true)**2/n
        guesses[np.where(concentrations==true)[0][0]].append(estimate[0])
    

    if ax_pred:
        cut = [27.5, 98.5]
        nudge = cut[1]-cut[0]
        ax_pred.set_xticks([0, 5, 10, 15, 20, 25, 100-nudge])
        ax_pred.set_xticklabels([0, 5, 10, 15, 20, 25, 100])
        lx = [np.min(test_y), cut[0], cut[1]-nudge, np.max(test_y)-nudge]
        ly = [np.min(test_y), cut[0], cut[1], np.max(test_y)]
        ax_pred.plot(lx[0:2], ly[0:2], color='black')
        ax_pred.plot(lx[2:4], ly[2:4], color='black')
        for i, g in guesses.items():
            true=concentrations[i]
            if true == 100:
                true -= nudge
            ax_pred.plot(true*np.ones_like(g), g, '.', color=c_colors[i], ms=5)
            ax_pred.plot(true, np.average(g), '_', color=c_colors[i], ms=20)
        ax_pred.text(0.04, 0.96, 'MSE = %.2f^2' % np.sqrt(mse), horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round", linewidth=2 ,ec=(0.6, 0.6, 0.6),fc=(0.95, 0.95, 0.95),), size='x-large', transform=ax_pred.transAxes)
    return mse


def draw_brakes(ax, x_pos, dx=0.005, dy=0.02):
    for a in [-0.005, 0.005]:
        pos = (x_pos + a)
        ax.plot([(pos-dx),(pos+dx)], [-dy,+dy], color='k', clip_on=False, transform=ax.transAxes)

def full_pls_eval():
    nrows=3
    fig_spectra, axs_spectra = plt.subplots(nrows=nrows, sharex='all', figsize=default_figsize)
    fig_pls, axs_pls = plt.subplots(nrows=nrows, sharex='all', figsize=default_figsize)

    pls_evaluation(lambdas*1e9, concentrations, train_set, test_set, axs_spectra[0], axs_pls[0])
    pls_evaluation(lambdas*1e9, concentrations, EMSC_corrected_train, EMSC_corrected_train, axs_spectra[1], axs_pls[1])
    pls_evaluation(lambdas*1e9, concentrations, optically_corrected_train, optically_corrected_train, axs_spectra[2], axs_pls[2])

    for i, title in enumerate(methods):
        axs_pls[i].set_ylabel('Predicted concentration [%]')
        axs_spectra[i].text(0.5, 0.96, title, horizontalalignment='center', verticalalignment='top', bbox=dict(boxstyle="round", linewidth=2, ec=set_colors[i],fc=(0.9, 0.9, 0.9),), size='x-large', transform=axs_spectra[i].transAxes)
        axs_pls[i].text(0.5, 0.96, f'PLSR based on {title[0].lower()}{title[1:]}', horizontalalignment='center', verticalalignment='top', bbox=dict(boxstyle="round", linewidth=2, ec=set_colors[i],fc=(0.9, 0.9, 0.9),), size='x-large', transform=axs_pls[i].transAxes)
    draw_brakes(axs_pls[i], 0.908)
    axs_spectra[0].set_ylabel('Absorbance [%]')
    axs_spectra[1].set_ylabel('Absorbance [%]')
    axs_spectra[2].set_ylabel('Absorption coefficient [m^-1]')
    axs_spectra[-1].set_xlabel('Wavelength [nm]')
    axs_pls[-1].set_xlabel('True concentration [%]')

    lines = [Line2D([0], [0], color=c_colors[i], lw=4) for i, c in enumerate(concentrations)]
    labels = ['%.1f' % c for c in concentrations]
    axs_spectra[1].legend(lines, labels, loc='upper center', title='Limestone concentration [%]', ncol=8, bbox_to_anchor=(0.5, 0.8))
    axs_pls[1].legend(lines, labels, loc='upper center', title='Limestone concentration [%]', ncol=8, bbox_to_anchor=(0.5, 0.8))
    fig_spectra.subplots_adjust(top=0.989,
                                bottom=0.041,
                                left=0.058,
                                right=0.988,
                                hspace=0.0,
                                wspace=0.2)
    fig_pls.subplots_adjust(top=0.989,
                            bottom=0.041,
                            left=0.057,
                            right=0.988,
                            hspace=0.0,
                            wspace=0.2)
    fig_spectra.savefig('spectral_results.svg')
    fig_pls.savefig('pls_results.svg')
    plt.show()


def single_sample_comparison(i):
    fig, axs = plt.subplots(nrows=3, sharex='all', figsize=default_figsize)
    cs = [0, 13, 15]
    for ax, c in zip(axs, cs):
        optically_corrected = optically_corrected_test[c][i]
        EMSC_corrected = EMSC_corrected_test[c][i]
        raw_data = test_set[c][i]
        ax2 = ax.twinx()
        ax.plot(lambdas*1e9, raw_data, label='raw_data', color=set_colors[0])
        ax.plot(lambdas*1e9, EMSC_corrected, label='EMSC_corrected', color=set_colors[1])
        ax2.plot(lambdas*1e9, optically_corrected, label='optically_corrected', color=set_colors[2])
        ax.text(0.35, 0.96, f'Limestone concentration: {concentrations[c]:.0f}%', horizontalalignment='center', verticalalignment='top', bbox=dict(boxstyle="round", linewidth=2, ec=c_colors[c],fc=(0.9, 0.9, 0.9),), size='x-large', transform=ax.transAxes)
        ax.set_ylabel('Absorbance[%]')
        ax2.set_ylabel('Absorption coefficient [m^-1]')
    ax.set_xlabel('Wavelength [nm]')
    lines = [Line2D([0], [0], color=color, lw=4) for color in set_colors]
    axs[1].legend(lines, methods, loc='upper center', bbox_to_anchor=(0.35, 0.8))
    fig.subplots_adjust(top=0.989,
                        bottom=0.043,
                        left=0.058,
                        right=0.942,
                        hspace=0.0,
                        wspace=0.117)
    fig.savefig('single_sample_results.svg')
    plt.show()

full_pls_eval()
single_sample_comparison(4)