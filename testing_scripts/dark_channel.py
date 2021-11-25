from matplotlib import cm, image
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

def dark_channel(image, patch_r):
    h, w, _ = image.shape
    min_channel_value = np.empty((h, w))
    for r in range(h):
        for c in range(w):
            min_channel_value[r, c] = np.min(image[r, c])
    j = np.empty((h, w))
    for r in range(h):
        for c in range(w):
            r_low = max(0, r-patch_r)
            r_high = min(h, r+patch_r)
            c_low = max(0, c-patch_r)
            c_high = min(w, c+patch_r)
            j[r, c] = np.min(min_channel_value[r_low:r_high,c_low:c_high])
    return j

def atmospheric_light(image, dark_channel):
    a = dark_channel.flatten()
    n = dark_channel.size
    k = int(n/100)
    brightest_in_j = np.argpartition(a, -k)[-k:]
    im_flat = image.reshape((n, 3))
    im_flat_intensity = np.sum(im_flat, axis=1)
    return im_flat[np.argmax(np.array([im_flat_intensity[i] for i in brightest_in_j]))]

def transmission(image, atmospheric_light, patch_r, w_haze):
    x = image/atmospheric_light
    h, w, _ = image.shape
    t = np.empty((h, w))
    for r in range(h):
        for c in range(w):
            r_low = max(0, r-patch_r)
            r_high = min(h, r+patch_r)
            c_low = max(0, c-patch_r)
            c_high = min(w, c+patch_r)
            t[r, c] = 1-w_haze*np.min(x[r_low:r_high,c_low:c_high])
    return t

def radiance(image, atmospheric_light, transmission, t_0):
    t = np.maximum(transmission, t_0)
    s = image-atmospheric_light
    s = np.transpose(s, axes=[2, 0, 1])
    s = (s/t).astype(np.dtype(np.int32))
    s = np.transpose(s, axes=[1, 2, 0])
    s = (s + atmospheric_light)
    s = s.astype(np.dtype(np.int32))
    return s

if __name__ == "__main__":

    im = image.imread('forrest_hallway.jpg')
    im = im.astype(np.dtype(np.int32))
    patch_r = 7
    w = 0.4
    t_0 = 0.2

    j = dark_channel(im, patch_r)
    print(np.average(j))
    a = atmospheric_light(im, j)
    t = transmission(im, a, patch_r, w)
    print(np.average(t))
    rad = radiance(im, a, t, t_0)
    j_aft = dark_channel(rad, patch_r)
    print(np.average(j_aft))
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(im)
    axs[1].imshow(t, cmap='gray')
    axs[2].imshow(rad)
    plt.show()