# include "fft.hpp"
# include <pybind11/pybind11.h>
# include <pybind11/stl.h>
# include <pybind11/operators.h>
# include <pybind11/complex.h>

namespace py = pybind11;

const double eps = 1e-4;

vector<Complex> fft1d(vector<Complex> &input) {
    cfft_plan plan = make_cfft_plan(input.size());

    // Convert input to double array that can be used in pocketfft
    double *data = new double[2 * input.size()];
    for (size_t i = 0; i < input.size(); i++) {
        data[2 * i] = input[i].real();
        data[2 * i + 1] = input[i].imag();
    }

    // Compute FFT
    cfft_forward(plan, data, 1.0);

    // Convert the result to Complex array
    vector<Complex> result(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        result[i] = Complex(data[2 * i], data[2 * i + 1]);
    }

    // Free memory
    delete[] data;
    destroy_cfft_plan(plan);

    return result;
}

vector<Complex> ifft1d(vector<Complex> &input) {
    cfft_plan plan = make_cfft_plan(input.size());

    // Convert input to double array that can be used in pocketfft
    double *data = new double[2 * input.size()];
    for (size_t i = 0; i < input.size(); i++) {
        data[2 * i] = input[i].real();
        data[2 * i + 1] = input[i].imag();
    }

    // Compute inverse FFT
    cfft_backward(plan, data, 1.0/input.size());

    // Convert the result to Complex array
    vector<Complex> result(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        result[i] = Complex(data[2 * i], data[2 * i + 1]);
    }

    // Free memory
    delete[] data;
    destroy_cfft_plan(plan);

    return result;
}

// 2D FFT
vector<vector<Complex>> fft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();

    vector<vector<Complex>> result(n, vector<Complex>(m));

    // FFT for each row
    for (int i = 0; i < n; i++) {
        result[i] = fft1d(input[i]);
    }

    // FFT for each column
    for (int i = 0; i < m; i++) {
        vector<Complex> column(n);
        for (int j = 0; j < n; j++) {
            column[j] = result[j][i];
        }
        column = fft1d(column);
        for (int j = 0; j < n; j++) {
            result[j][i] = column[j];
        }
    }

    return result;
}

vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();
    vector<vector<Complex>> result(n, vector<Complex>(m));

    // FFT for each row
    for (int i = 0; i < n; i++) {
        result[i] = ifft1d(input[i]);
    }

    // FFT for each column
    for (int i = 0; i < m; i++) {
        vector<Complex> column(n);
        for (int j = 0; j < n; j++) {
            column[j] = result[j][i];
        }
        column = ifft1d(column);
        for (int j = 0; j < n; j++) {
            result[j][i] = column[j];
        }
    }

    return result;
}

vector<vector<double>> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel) {
    int n = input.size(), m = input[0].size();
    int n_kernel = kernel.size(), m_kernel = kernel[0].size();

    // Preprocess the kernel
    kernel = paddingKernel(kernel, n, m);  // Padding zero to make the size of kernel to be the same as input
    kernel = roll2d(kernel, -n_kernel / 2, -m_kernel / 2);  // Roll the kernel to the center

    // Represent input and kernel as complex number
    vector<vector<Complex>> input_complex(n, vector<Complex>(m));
    vector<vector<Complex>> kernel_complex(n, vector<Complex>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            input_complex[i][j] = input[i][j];
            kernel_complex[i][j] = kernel[i][j];
        }
    }

    // Compute FFT for input and kernel
    vector<vector<Complex>> input_fft = fft2d(input_complex);
    vector<vector<Complex>> kernel_fft = fft2d(kernel_complex);

    // Compute the convolution in frequency domain (multiplication)
    vector<vector<Complex>> result(n, vector<Complex>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i][j] = input_fft[i][j] * kernel_fft[i][j];
        }
    }

    // Compute the inverse FFT
    vector<vector<Complex>> result_complex = ifft2d(result);

    // Calculate the magnitude of the result
    vector<vector<double>> result_magnitude(n, vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result_magnitude[i][j] = abs(result_complex[i][j]);
        }
    }

    return result_magnitude;
}

vector<vector<double>> paddingKernel(vector<vector<double>> &kernel, int n, int m) {
    int n_kernel = kernel.size(), m_kernel = kernel[0].size();
    vector<vector<double>> result(n, vector<double>(m, 0));
    for (int i = 0; i < n_kernel; i++) {
        for (int j = 0; j < m_kernel; j++) {
            result[i][j] = kernel[i][j];
        }
    }
    return result;
}

vector<vector<double>> roll2d(vector<vector<double>> &input, int shift_x, int shift_y) {
    int n = input.size(), m = input[0].size();
    vector<vector<double>> result(n, vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[(i + shift_x + n) % n][(j + shift_y + m) % m] = input[i][j];
        }
    }
    return result;
}

PYBIND11_MODULE(fft, m) {
    m.doc() = "Fast Fourier Transform"; // optional module docstring

    m.def("fft1d", &fft1d, "Fast Fourier Transform for 1D signal.");
    m.def("ifft1d", &ifft1d, "Inverse Fast Fourier Transform for 1D signal.");
    m.def("fft2d", &fft2d, "Fast Fourier Transform for 2D signal.");
    m.def("ifft2d", &ifft2d, "Inverse Fast Fourier Transform for 2D signal.");
    m.def("fftconvolve2d", &fftconvolve2d, "2D convolution using FFT.");
}
