import matplotlib
from matplotlib import pyplot as plt
import spectral as sp
import numpy as np
from PIL import Image
from graph_scraper import scrape
from testing_scripts.dark_channel import transmission

def save_as_rgb(header, image, filename):
    img = sp.envi.open(header, image)
    rgb_img = img[:,:,(50,130,220)]
    rgb_img = rgb_img/np.max(rgb_img)
    matplotlib.image.imsave(filename, rgb_img)

def read_data():
    header_paths = ['neo_data\Tray1_SWIR_384_SN3154_6056us_2021-04-08T154417_raw_rad_ref_float32.hdr', 'neo_data\Tray2_SWIR_384_SN3154_6056us_2021-04-08T155241_raw_rad_ref_float32.hdr']
    image_paths = ['neo_data\Tray1_SWIR_384_SN3154_6056us_2021-04-08T154417_raw_rad_ref_float32.img', 'neo_data\Tray2_SWIR_384_SN3154_6056us_2021-04-08T155241_raw_rad_ref_float32.img']
    roi_mask_paths = ['neo_data\\Tray1_mask.png', 'neo_data\\Tray2_mask.png']

    mask_value_to_consentration = {
        1: 0.0,
        5: 0.5,
        10: 1.0,
        20: 2.0,
        30: 3.0,
        40: 4.0,
        50: 5.0,
        60: 6.0,
        70: 7.0,
        80: 8.0,
        90: 9.0,
        100: 10.0,
        150: 15.0,
        200: 20.0,
        250: 25.0,
        255: 100.0
    }

    rois = [None]*len(mask_value_to_consentration)

    for i in range(len(header_paths)):
        img = sp.envi.open(header_paths[i], image_paths[i])

        white_panel = img[1500:,:,:]
        white_panel_spectrum = np.average(white_panel, axis=(0,1))

        roi_mask_rgba = np.array(Image.open(roi_mask_paths[i]))
        roi_mask = roi_mask_rgba[:,:,0]

        for j, mask_value in enumerate(mask_value_to_consentration.keys()):
            corners = np.where(roi_mask == mask_value)
            if len(corners[0]):
                rois[j] = 0.5*img[corners[0][0]:corners[0][1], corners[1][0]:corners[1][1], :]/white_panel_spectrum

    return 1e-9*np.array(img.bands.centers), rois, np.array(list(mask_value_to_consentration.values()))


def get_reflectance(frame_n=None, frame_m=None):
    band_centers, rois, concentrations = read_data()
    reflectance = [None]*len(rois)
    for i, roi in enumerate(rois):
        frame_n = frame_n if frame_n else roi.shape[0]
        frame_m = frame_m if frame_m else roi.shape[1]
        n = int(roi.shape[0]/frame_n)
        m = int(roi.shape[1]/frame_m)
        reflectance[i] = []
        for x in range(n):
            for y in range(m):
                reflectance[i].append(np.average(roi[frame_n*x:frame_n*(x+1),frame_m*y:frame_m*(y+1)], axis=(0,1)))
        reflectance[i] = np.array(reflectance[i])
    return band_centers, reflectance, concentrations

def get_transmission(frame_n=None, frame_m=None):
    band_centers, reflectance, concentrations = get_reflectance(frame_n, frame_m)
    transmission = [1-r for r in reflectance]
    return band_centers, transmission, concentrations


def show_data_in_rgb():
    band_centers, rois, concentrations = read_data()
    fig, axs = plt.subplots(len(rois), figsize=(8,12))
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 2.0   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    max_value = np.max(np.array([np.max(roi.flatten()) for roi in rois]))
    for i, roi in enumerate(rois):
        axs[i].set_title('Concentration: %.2f' % concentrations[i])
        rgb = roi[:,:,(50,130,220)]/max_value
        axs[i].imshow(rgb)
    plt.show()


def get_absorbance_data(lambdas):
    x_sio2, y_sio2 = scrape('sio2_absorbance_scrape.png', 10000, 4000, 0, 0.1)
    x_caco3, y_caco3 = scrape('caco3_absorbance_scrape.png', 10000, 4000, 0, 0.2)

    absorbance_sio2 = np.interp(1/(100*lambdas), np.flip(x_sio2), np.flip(y_sio2))
    absorbance_caco3 = np.interp(1/(100*lambdas), np.flip(x_caco3), np.flip(y_caco3))

    return absorbance_sio2, absorbance_caco3


if __name__ == "__main__":
    lambdas, transmission, concentrations = get_transmission(100, 100)
    absorbance_sio2, absorbance_caco3 = get_absorbance_data(lambdas)

    fig, axs = plt.subplots(nrows=2, figsize=(12,5), sharex=True)
    ax, ax2 = axs
    ax.plot(lambdas, transmission[-1][0], label='Transmission: Limestone')
    ax.plot(lambdas, transmission[0][0], label='Transmission: Sand')
    ax2.plot(lambdas, absorbance_caco3, label='CaCO3 absorbance')
    ax2.plot(lambdas, absorbance_sio2, label='SiO2 absorbance')
    ax.legend()
    ax2.legend()
    plt.xlabel('Wavelength')
    plt.show()
