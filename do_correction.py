from EMSC import binary_EMSC
from read_data import get_transmission
from read_data import get_absorbance_data
import numpy as np
from random import shuffle
import pickle
from reflectance_model import fit_gamma_model


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

EMSC_corrected_train, EMSC_corrected_test = apply_func_on_sets(train_set, test_set, lambda s : binary_EMSC(lambdas, ref_0, ref_100, s)[0])

optically_corrected_train, optically_corrected_test = apply_func_on_sets(train_set, test_set, lambda s : fit_gamma_model(lambdas, 1-s)[0])

sets = {'lambdas': lambdas,
        'concentrations': concentrations,
        'train_set': train_set,
        'test_set': test_set,
        'EMSC_corrected_train': EMSC_corrected_train,
        'EMSC_corrected_test': EMSC_corrected_test,
        'optically_corrected_train': optically_corrected_train,
        'optically_corrected_test': optically_corrected_test,
        'ref_0': ref_0,
        'ref_100': ref_100}

with open('results.pickle', 'wb') as output_file:
    pickle.dump(sets, output_file)