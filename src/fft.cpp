# include "fft.hpp"
# include <pybind11/pybind11.h>
# include <pybind11/stl.h>
# include <pybind11/operators.h>
# include <pybind11/complex.h>

namespace py = pybind11;

vector<Complex> fft1d(vector<Complex> &input) {
    int n = input.size();
    if (n == 1) {
        return input;
    }

    // padding zero to make the size of input to be power of 2
    while (n & (n - 1)) {
        input.emplace_back(0);
        n++;
    }

    // Divide into even and odd function
    vector<Complex> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; i++) {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }
    
    // Recursively compute FFT
    vector<Complex> even_result = fft1d(even);
    vector<Complex> odd_result = fft1d(odd);

    // Merge
    vector<Complex> result(n);
    double angle = 2 * M_PI / n;
    Complex omega(1, 0), omega_n(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; i++) {
        result[i] = even_result[i] + omega * odd_result[i];
        result[i + n / 2] = even_result[i] - omega * odd_result[i];
        omega *= omega_n;
    }
    return result;
}

vector<Complex> ifft1d(vector<Complex> &input, bool root) {
    int n = input.size();
    if (n == 1) {
        return input;
    }

    // Divide into even and odd function
    vector<Complex> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; i++) {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }

    // Recursively compute FFT
    vector<Complex> even_result = ifft1d(even, false);
    vector<Complex> odd_result = ifft1d(odd, false);

    // Merge
    vector<Complex> result(n);
    double angle = 2 * M_PI / n;
    Complex omega(1, 0), omega_n(cos(angle), sin(angle));
    omega_n = 1.0 / omega_n;

    for (int i = 0; i < n / 2; i++) {
        result[i] = even_result[i] + omega * odd_result[i];
        result[i + n / 2] = even_result[i] - omega * odd_result[i];
        omega *= omega_n;
    }

    if (root) {
        for (int i = 0; i < n; i++) {
            result[i] /= n;
        }
    }

    return result;
}

// 2D FFT
vector<vector<Complex>> fft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();

    // Padding zero to make the size of input to be power of 2
    while (m & (m - 1)) {
        for (int i = 0; i < n; i++) {
            input[i].emplace_back(0);
        }
        m++;
    }
    while (n & (n - 1)) {
        input.emplace_back(vector<Complex>(m, 0));
        n++;
    }

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
    if (n & (n - 1) || m & (m - 1)) {
        throw std::invalid_argument("The size of input should be power of 2.");
    }

    // Let the size of kernel be the same as input
    kernel = paddingKernel(kernel, n, m);

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

PYBIND11_MODULE(fft, m) {
    m.doc() = "Fast Fourier Transform"; // optional module docstring

    m.def("fft1d", &fft1d, "Fast Fourier Transform for 1D signal.");
    m.def("ifft1d", &ifft1d, "Inverse Fast Fourier Transform for 1D signal.");
    m.def("fft2d", &fft2d, "Fast Fourier Transform for 2D signal.");
    m.def("ifft2d", &ifft2d, "Inverse Fast Fourier Transform for 2D signal.");
    m.def("fftconvolve2d", &fftconvolve2d, "2D convolution using FFT.");
}
