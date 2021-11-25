from numpy.lib.type_check import imag
from distributions import gaussian, zero_mean_uncorrelated_gaussian
import scipy
from scipy import signal, misc
import matplotlib.pyplot as plt
import numpy as np


def blur(image, sigma):
    x, y, p = zero_mean_uncorrelated_gaussian(sigma)
    blurred_channels = np.array([signal.convolve2d(image[:,:,channel_index], p) for channel_index in range(3)], dtype=np.dtype(np.int32))
    blurred_image = np.transpose(blurred_channels, axes=[1, 2, 0])
    return blurred_image


image = misc.face()

sigmas = [10, 6, 16, 30]

images = [image] + [None]*len(sigmas)
fig, axs = plt.subplots(1, len(images))
axs[0].imshow(image)

for i, sigma in enumerate(sigmas):
    images[i+1] = blur(images[i], sigma)
    axs[i+1].imshow(images[i+1])

plt.show()