import numpy as np

def get_spectrum(image):
    spectrum = np.fft.fft2(image)
    spectrum = np.fft.fftshift(spectrum)
    return np.abs(spectrum)

def grayvalue_image(image):
    return np.sum(image, axis=-3, keepdims=True)
