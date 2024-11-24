import os
import time
import fft
import numpy as np
import scipy.signal
import pytest

def test_conv_speed():
    img_sizes = [256, 1024, 4096]
    kernel_sizes = [15, 25, 35]

    filename = os.path.join('assets', 'performance_conv.txt')
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        print("====== Running convolution speed test ======")

        header = f"{'image size':<15} {'kernel size':<15} {'fft':<15} {'scipy':<15} {'speedup':<15}"
        print(header)
        f.write(header + "\n")

        for img_size in img_sizes:
            for kernel_size in kernel_sizes:
                img = np.random.rand(img_size, img_size)
                kernel = np.random.rand(kernel_size, kernel_size)

                start_fft = time.time()
                result_fft = fft.fftconvolve2d(img, kernel)
                time_fft = time.time() - start_fft

                start_scipy = time.time()
                result_scipy = scipy.signal.convolve2d(img, kernel, mode='same')
                time_scipy = time.time() - start_scipy

                speedup = time_scipy / time_fft
                
                output = f"{img_size}x{img_size:<12} {kernel_size}x{kernel_size:<12} {time_fft:<15.6f} {time_scipy:<15.6f} {speedup:<15.6f}"
                print(output)
                f.write(output + "\n")

test_conv_speed()
