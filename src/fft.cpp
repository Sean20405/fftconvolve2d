# include "fft.hpp"
# include <pybind11/pybind11.h>
# include <pybind11/stl.h>
# include <pybind11/operators.h>
# include <pybind11/complex.h>

# include <sys/time.h>

namespace py = pybind11;

// Choose which FFT implementation to use
vector<Complex> fft1d(vector<Complex> &input, string method) {
    if (method == "mixed_radix") return MixedRadixFFT::fft1d(input);
    else if (method == "cooley_tukey") return CooleyTukeyFFT::fft1d(input); 
    else throw invalid_argument("Invalid method");
}

vector<Complex> ifft1d(vector<Complex> &input, string method) {
    if (method == "mixed_radix") return MixedRadixFFT::ifft1d(input);
    else if (method == "cooley_tukey") return CooleyTukeyFFT::ifft1d(input); 
    else throw invalid_argument("Invalid method");

}

vector<vector<Complex>> fft2d(vector<vector<Complex>> &input, string method) {
    if (method == "mixed_radix") return MixedRadixFFT::fft2d(input);
    else if (method == "cooley_tukey") return CooleyTukeyFFT::fft2d(input); 
    else throw invalid_argument("Invalid method");

}
vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input, string method) {
    if (method == "mixed_radix") return MixedRadixFFT::ifft2d(input);
    else if (method == "cooley_tukey") return CooleyTukeyFFT::ifft2d(input); 
    else throw invalid_argument("Invalid method");

}

vector<vector<double>> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel, string method) {
    struct timeval start, end, start2, end2;

    int n = input.size(), m = input[0].size();
    int n_kernel = kernel.size(), m_kernel = kernel[0].size();

    // Preprocess the image and kernel
    cout << "Preprocessing the image and kernel" << endl;
    cout << "    Padding to faster size";
    gettimeofday(&start2, 0);
    int n_fast = *lower_bound(fastSize.begin(), fastSize.end(), n + n_kernel - 1), m_fast = *lower_bound(fastSize.begin(), fastSize.end(), m + m_kernel - 1);
    paddingInput(input, n_fast, m_fast);
    gettimeofday(&end2, 0);
    cout << " - " << (end2.tv_sec - start2.tv_sec) + (end2.tv_usec - start2.tv_usec) / 1e6 << "s" << endl;

    cout << "    Padding kernel";
    gettimeofday(&start2, 0);
    paddingKernel(kernel, n_fast, m_fast);  // Padding zero to make the size of kernel to be the same as input
    gettimeofday(&end2, 0);
    cout << " - " << (end2.tv_sec - start2.tv_sec) + (end2.tv_usec - start2.tv_usec) / 1e6 << "s" << endl;
    kernel = roll2d(kernel, -n_kernel / 2, -m_kernel / 2);  // Roll the kernel to the center

    // Represent input and kernel as complex number
    vector<vector<Complex>> input_complex(n_fast, vector<Complex>(m_fast));
    vector<vector<Complex>> kernel_complex(n_fast, vector<Complex>(m_fast));
    for (int i = 0; i < n_fast; i++) {
        for (int j = 0; j < m_fast; j++) {
            input_complex[i][j] = input[i][j];
            kernel_complex[i][j] = kernel[i][j];
        }
    }

    // Compute FFT for input and kernel
    cout << "Computing FFT for input and kernel" << endl;
    gettimeofday(&start, 0);
    vector<vector<Complex>> input_fft = fft2d(input_complex, method);
    vector<vector<Complex>> kernel_fft = fft2d(kernel_complex, method);
    gettimeofday(&end, 0);
    cout << " > " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // Compute the convolution in frequency domain (multiplication)
    cout << "Computing the convolution in frequency domain";
    gettimeofday(&start, 0);
    vector<vector<Complex>> result(n_fast, vector<Complex>(m_fast));
    for (int i = 0; i < n_fast; i++) {
        for (int j = 0; j < m_fast; j++) {
            result[i][j] = input_fft[i][j] * kernel_fft[i][j];
        }
    }
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // Compute the inverse FFT
    cout << "Computing the inverse FFT" << endl;
    gettimeofday(&start, 0);
    vector<vector<Complex>> result_complex = ifft2d(result, method);
    gettimeofday(&end, 0);
    cout << " > " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // Calculate the magnitude of the result
    cout << "Calculating the magnitude of the result";
    gettimeofday(&start, 0);
    vector<vector<double>> result_magnitude(n_fast, vector<double>(m_fast));
    for (int i = 0; i < n_fast; i++) {
        for (int j = 0; j < m_fast; j++) {
            result_magnitude[i][j] = abs(result_complex[i][j]);
        }
    }
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // Crop the center of the result
    cout << "Cropping the center of the result";
    gettimeofday(&start, 0);
    vector<vector<double>> result_magnitude_cropped(n, vector<double>(m));
    int n_padding_size = (n_fast - n) / 2, m_padding_size = (m_fast - m) / 2;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result_magnitude_cropped[i][j] = result_magnitude[i + n_padding_size][j + m_padding_size];
        }
    }
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl << endl;

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

    m.def("fft1d", &fft1d, "Fast Fourier Transform for 1D signal.",
        py::arg("input"), py::arg("method")="mixed_radix");
    m.def("ifft1d", &ifft1d, "Inverse Fast Fourier Transform for 1D signal.",
        py::arg("input"), py::arg("method")="mixed_radix");
    m.def("fft2d", &fft2d, "Fast Fourier Transform for 2D signal.",
        py::arg("input"), py::arg("method")="mixed_radix");
    m.def("ifft2d", &ifft2d, "Inverse Fast Fourier Transform for 2D signal.",
        py::arg("input"), py::arg("method")="mixed_radix");
    m.def("fftconvolve2d", &fftconvolve2d, "2D convolution using FFT.",
        py::arg("input"), py::arg("kernel"), py::arg("method")="mixed_radix");
    m.def("paddingInput", &paddingInput, "Padding zero at the surrounding of the input until the size of input is n x m.");
}
