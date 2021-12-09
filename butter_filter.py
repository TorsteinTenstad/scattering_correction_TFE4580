
from scipy.signal import butter, lfilter, freqz
import numpy as np

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    mean=np.average(data)
    y = lfilter(b, a, data-mean)+mean
    return y