# fftconvolve2d - FFT Convolution Operations for Image Processing
A FFT version of [`scipy.signal.convolve2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html). It’s compatible to numpy array so that user can directly switch from `convolve2d` to it.

## Introduction

This project implements 2D convolution using Fast Fourier Transform (FFT). It includes multiple FFT implementation methods, such as mixed-radix FFT and Cooley-Tukey FFT, and provides Python bindings for easy use in Python.

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/Sean20405/fftconvolve2d
    cd fftconvolve2d
    ```

2. Install Python dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Compile the C++ code:
    ```sh
    make
    ```

## Usage
After compiling, you can import and use the FFT module in Python:

```python
import fft

# Prepare input data and kernel
input_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
kernel = [[1.0, 0.0], [0.0, -1.0]]

# Perform 2D convolution using FFT
result = fft.fftconvolve2d(input_data, kernel, method="mixed_radix", mode="full")
```

> [!CAUTION]
> 
> The `.so` file must be placed in the same directory as the Python file.

## API Description

```python
fft.fftconvolve2d(input, kernel, method='mixed_radix', mode='full')
```

Performs 2D convolution using FFT.

- Parameters:
    - `input`, `kernel` (list or `numpy.ndarray`): The input and the convolution kernel, both dimension should be 2.
    - `method` ({`'mixed_radix'`, `'cooley_tukey'`}, optional): The FFT algorithm to use. Deafult is `'mixed_radix'`
    - `mode` ({`'full'`, `'valid'`, `'same'`}, optional): The convolution mode. More to see [`scipy.signal.
convolve2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)
- Returns
    - `numpy.ndarray`: The result of the convolution.

```python
fft.fft2d(input, method='mixed_radix')
```

Performs 2D FFT.

- Parameters:
    - `input` (list): The dimension should be 2.
    - `method` ({`'mixed_radix'`, `'cooley_tukey'`}, optional): The FFT algorithm to use. Deafult is `'mixed_radix'`
- Returns
    - list: The result of the 2D FFT.

```python
fft.ifft2d(input, method='mixed_radix')
```

Performs 2D inverse FFT.

- Parameters:
    - `input` (list): The dimension should be 2.
    - `method` ({`'mixed_radix'`, `'cooley_tukey'`}, optional): The FFT algorithm to use. Deafult is `'mixed_radix'`
- Returns
    - list: The result of the 2D inverse FFT.

For more example about `fft.fft2d()` and `fft.ifft2d()`, see [tests/test_fft.py](tests/test_fft.py)


## File Structure
```
fftconvolve2d/
├── .github/workflows  // CI
├── assets             // Performance, result picture, ...
├── src/
│   ├── pocketfft/
│   ├── fft_impl.cpp   // Detail implementation of FFT
│   ├── fft_impl.hpp
│   ├── fft.cpp        // fftconvolve2d() + pybind11
│   ├── fft.hpp
│   ├── utils.cpp      // e.g. padding, cropping, ...
│   └── utils.hpp
├── tests/             // pytest
├── Makefile
└── (... other files)
```

## Test

You can run the tests using pytest:

```sh
make test
```

## Acknowledgement

Thanks for providing [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) that serve a fast mixed-radix FFT algorithm.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.