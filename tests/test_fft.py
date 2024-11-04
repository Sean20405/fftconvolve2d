import fft
import numpy as np
import random
import pytest

def test_1d():
    length = 16
    x = np.array([random.uniform(-length, length) for _ in range(length)])
    x_fft = fft.fft1d(x)
    x_ifft = fft.ifft1d(x_fft)

    assert np.array(x_ifft) == pytest.approx(x)

def test_2d():
    height, width = 16, 16
    x = np.array([[random.uniform(-width, width) for _ in range(width)] for _ in range(height)])
    x_fft = fft.fft2d(x)
    x_ifft = fft.ifft2d(x_fft)

    assert np.array(x_ifft) == pytest.approx(x)