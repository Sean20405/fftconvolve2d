import scipy
import fft
import numpy as np
import random
import pytest

def test_correctness():
    x_size = 512
    kernel_size = 15

    x = np.array([[random.randint(0, 255) for _ in range(x_size)] for _ in range(x_size)])  # image-like
    kernel = np.array([[random.uniform(0, 1) for _ in range(kernel_size)] for _ in range(kernel_size)])

    x_fft = fft.fftconvolve2d(x, kernel, mode='same')
    x_scipy = scipy.signal.convolve2d(x, kernel, mode='same')

    assert np.array(x_fft) == pytest.approx(x_scipy)

def test_mode():
    x_size = 512
    kernel_size = 15

    x = np.array([[random.randint(0, 255) for _ in range(x_size)] for _ in range(x_size)])  # image-like
    kernel = np.array([[random.uniform(0, 1) for _ in range(kernel_size)] for _ in range(kernel_size)])

    modes = ['full', 'valid', 'same']
    for mode in modes:
        x_fft = fft.fftconvolve2d(x, kernel, mode=mode)
        x_scipy = scipy.signal.convolve2d(x, kernel, mode=mode)

        assert np.array(x_fft) == pytest.approx(x_scipy)

def test_array():
    x_size = 32
    kernel_size = 7

    x = [[random.randint(0, 255) for _ in range(x_size)] for _ in range(x_size)]
    kernel = [[random.uniform(0, 1) for _ in range(kernel_size)] for _ in range(kernel_size)]
    x_np = np.array(x)
    kernel_np = np.array(kernel)

    x_fft = fft.fftconvolve2d(x, kernel, mode='same')
    x_np_fft = fft.fftconvolve2d(x_np, kernel_np, mode='same')
    x_scipy = scipy.signal.convolve2d(x_np, kernel_np, mode='same')

    assert np.array(x_fft) == pytest.approx(x_scipy)
    assert x_scipy == pytest.approx(x_np_fft)

def test_method():
    x_size = 512
    kernel_size = 15

    x = np.array([[random.randint(0, 255) for _ in range(x_size)] for _ in range(x_size)])  # image-like
    kernel = np.array([[random.uniform(0, 1) for _ in range(kernel_size)] for _ in range(kernel_size)])

    x_scipy = scipy.signal.convolve2d(x, kernel, mode='same')

    methods = ['cooley_tukey', 'mixed_radix']
    for method in methods:
        x_fft = fft.fftconvolve2d(x, kernel, mode='same', method=method)
        assert np.array(x_fft) == pytest.approx(x_scipy)