import scipy
import fft
import numpy as np
import random
import pytest

def test_conv():
    x_size = 256
    kernel_size = 5

    x = np.array([[random.randint(0, 255) for _ in range(x_size)] for _ in range(x_size)])  # image-like
    kernel = np.array([[random.uniform(0, 1) for _ in range(kernel_size)] for _ in range(kernel_size)])
    x_conv = fft.fftconvolve2d(x, kernel)

    assert np.array(x_conv).shape == (x_size, x_size)
    assert np.array(x_conv).dtype == np.float64

def test_correctness():
    x_size = 512
    kernel_size = 15

    x = np.array([[random.randint(0, 255) for _ in range(x_size)] for _ in range(x_size)])  # image-like
    kernel = np.array([[random.uniform(0, 1) for _ in range(kernel_size)] for _ in range(kernel_size)])

    x_fft = fft.fftconvolve2d(x, kernel)
    x_scipy = scipy.signal.convolve2d(x, kernel, mode='same')

    assert np.array(x_fft) == pytest.approx(x_scipy)
