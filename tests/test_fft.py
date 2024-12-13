import fft
import numpy as np
import random
import pytest

def test_correctness():
    height, width = 16, 16
    x = np.array([[random.uniform(-width, width) for _ in range(width)] for _ in range(height)])

    x_np_fft = np.fft.fft2(x)
    methods = ['cooley_tukey', 'mixed_radix']
    for method in methods:
        x_fft = fft.fft2d(x, method=method)
        assert np.array(x_fft) == pytest.approx(x_np_fft)

def test_size():
    heights = [16, 64, 256]
    widths = [16, 64, 256]

    for height, width in zip(heights, widths):
        x = np.array([[random.uniform(-width, width) for _ in range(width)] for _ in range(height)])

        x_np_fft = np.fft.fft2(x)
        methods = ['cooley_tukey', 'mixed_radix']
        for method in methods:
            x_fft = fft.fft2d(x, method=method)
            assert np.array(x_fft) == pytest.approx(x_np_fft)

def test_inverse():
    height, width = 16, 16
    x = np.array([[random.uniform(-width, width) for _ in range(width)] for _ in range(height)])

    methods = ['cooley_tukey', 'mixed_radix']
    for method in methods:
        x_fft = fft.fft2d(x, method=method)
        x_ifft = fft.ifft2d(x_fft, method=method)
        assert np.array(x_ifft) == pytest.approx(x)