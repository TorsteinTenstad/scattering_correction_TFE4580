import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from butter_filter import butter_lowpass_filter


def scrape(image_path, x_min, x_max, y_min, y_max):
    im = image.imread(image_path)
    grascale = np.average(im, axis=2)
    columns = grascale.T
    max_value = columns.shape[1]
    y_value = np.array([np.dot(np.flip(c), np.arange(len(c))/np.sum(c)) for c in columns])
    y0 = y_value[0]
    filtered = butter_lowpass_filter(y_value-y0, 0.1, 1)+y0
    scaled = (filtered-y_min)*(y_max-y_min)/max_value
    x = np.linspace(x_min, x_max, len(scaled))
    return x, scaled


if __name__ == "__main__":
    fig, ax = plt.subplots()
    x_sio2, y_sio2 = scrape('sio2_absorbance_scrape.png', 10000, 4000, 0, 0.1)
    x_caco3, y_caco3 = scrape('caco3_absorbance_scrape.png', 10000, 4000, 0, 0.2)
    ax.plot(x_sio2, y_sio2)
    ax.plot(x_caco3, y_caco3)
    ax.invert_xaxis()
    plt.ylim([0, 0.2])
    plt.show()