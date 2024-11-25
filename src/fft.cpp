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

    // FFT for each row  TODO: bottleneck
    for (int i = 0; i < n; i++) {
        result[i] = fft1d(input[i]);
    }

    // FFT for each column  TODO: bottleneck
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
    int output_n = n + n_kernel - 1, output_m = m + m_kernel - 1;

    // Preprocess the image and kernel
    cout << "Preprocessing the image and kernel" << endl;
    paddingInput(input, output_n, output_m);  // Padding zero to make the size of input to be the same as kernel
    paddingKernel(kernel, output_n, output_m);  // Padding zero to make the size of kernel to be the same as input
    kernel = roll2d(kernel, -n_kernel / 2, -m_kernel / 2);  // Roll the kernel to the center

    // Represent input and kernel as complex number
    vector<vector<Complex>> input_complex(output_n, vector<Complex>(output_m));
    vector<vector<Complex>> kernel_complex(output_n, vector<Complex>(output_m));
    for (int i = 0; i < output_n; i++) {
        for (int j = 0; j < output_m; j++) {
            input_complex[i][j] = input[i][j];
            kernel_complex[i][j] = kernel[i][j];
        }
    }

    // Compute FFT for input and kernel
    cout << "Computing FFT for input and kernel" << endl;
    vector<vector<Complex>> input_fft = fft2d(input_complex);
    vector<vector<Complex>> kernel_fft = fft2d(kernel_complex);

    // Compute the convolution in frequency domain (multiplication)
    cout << "Computing the convolution in frequency domain" << endl;
    vector<vector<Complex>> result(output_n, vector<Complex>(output_m));
    for (int i = 0; i < output_n; i++) {
        for (int j = 0; j < output_m; j++) {
            result[i][j] = input_fft[i][j] * kernel_fft[i][j];
        }
    }

    // Compute the inverse FFT
    cout << "Computing the inverse FFT" << endl;
    vector<vector<Complex>> result_complex = ifft2d(result);

    // Calculate the magnitude of the result
    cout << "Calculating the magnitude of the result" << endl;
    vector<vector<double>> result_magnitude(output_n, vector<double>(output_m));
    for (int i = 0; i < output_n; i++) {
        for (int j = 0; j < output_m; j++) {
            result_magnitude[i][j] = abs(result_complex[i][j]);
        }
    }

    // Crop the center of the result
    cout << "Cropping the center of the result" << endl << endl;
    vector<vector<double>> result_magnitude_cropped(n, vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result_magnitude_cropped[i][j] = result_magnitude[i + n_kernel / 2][j + m_kernel / 2];
        }
    }
    // return result_magnitude;
    return result_magnitude_cropped;
}

void paddingKernel(vector<vector<double>> &kernel, int n, int m) {
    int n_kernel = kernel.size();
    
    for (int i=0; i<n_kernel; i++) kernel[i].resize(m, 0);
    kernel.resize(n, vector<double>(m, 0));
    
    return;
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

// Padding zero at the surrounding of the input until the size of input is n x m
void paddingInput(vector<vector<double>> &input, int n, int m) {
    int n_input = input.size(), m_input = input[0].size();
    int n_padding_size = (n - n_input) / 2, m_padding_size = (m - m_input) / 2;

    for (int i=0; i<n_input; i++) {
        input[i].resize(m);
        for (int j=0; j<m_padding_size; j++) {
            input[i].insert(input[i].begin(), 0);
            input[i].push_back(0);
        }
    }

    input.resize(n, vector<double>(m));
    for (int i=0; i<n_padding_size; i++) {
        vector<double> padding(m, 0);
        input.insert(input.begin(), padding);
        input.push_back(padding);
    }
}

PYBIND11_MODULE(fft, m) {
    m.doc() = "Fast Fourier Transform"; // optional module docstring

    m.def("fft1d", &fft1d, "Fast Fourier Transform for 1D signal.");
    m.def("ifft1d", &ifft1d, "Inverse Fast Fourier Transform for 1D signal.");
    m.def("fft2d", &fft2d, "Fast Fourier Transform for 2D signal.");
    m.def("ifft2d", &ifft2d, "Inverse Fast Fourier Transform for 2D signal.");
    m.def("fftconvolve2d", &fftconvolve2d, "2D convolution using FFT.");
    m.def("paddingInput", &paddingInput, "Padding zero at the surrounding of the input until the size of input is n x m.");
}
