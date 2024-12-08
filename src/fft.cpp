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
    else if (method == "cooley_tukey") return CooleyTukeyFFT_MP::fft1d(input, input.size()); 
    else throw invalid_argument("Invalid method");
}

vector<Complex> ifft1d(vector<Complex> &input, string method) {
    if (method == "mixed_radix") return MixedRadixFFT::ifft1d(input);
    else if (method == "cooley_tukey") return CooleyTukeyFFT_MP::ifft1d(input, input.size()); 
    else throw invalid_argument("Invalid method");

}

vector<vector<Complex>> fft2d(vector<vector<Complex>> &input, string method) {
    if (method == "mixed_radix") return MixedRadixFFT::fft2d(input);
    else if (method == "cooley_tukey") return CooleyTukeyFFT_MP::fft2d(input); 
    else throw invalid_argument("Invalid method");

}
vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input, string method) {
    if (method == "mixed_radix") return MixedRadixFFT::ifft2d(input);
    else if (method == "cooley_tukey") return CooleyTukeyFFT_MP::ifft2d(input); 
    else throw invalid_argument("Invalid method");

}

vector<vector<double>> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel, string method, string mode) {
    struct timeval start, end, start2, end2;

    int n = input.size(), m = input[0].size();
    int n_kernel = kernel.size(), m_kernel = kernel[0].size();
    int n_fast, m_fast;

    // Preprocess the image and kernel
    cout << "Preprocessing the image and kernel" << endl;
    cout << "    Padding to faster size";
    gettimeofday(&start2, 0);
    std::tie(n_fast, m_fast) = paddingInput(input, n + n_kernel - 1, m + m_kernel - 1, method);
    gettimeofday(&end2, 0);
    cout << "    > " << (end2.tv_sec - start2.tv_sec) + (end2.tv_usec - start2.tv_usec) / 1e6 << "s" << endl;

    cout << "    Padding kernel";
    gettimeofday(&start2, 0);
    paddingKernel(kernel, n_fast, m_fast);  // Padding zero to make the size of kernel to be the same as input
    gettimeofday(&end2, 0);
    cout << " - " << (end2.tv_sec - start2.tv_sec) + (end2.tv_usec - start2.tv_usec) / 1e6 << "s" << endl;
    kernel = roll2d(kernel, -n_kernel / 2, -m_kernel / 2);  // Roll the kernel to the center

    // Represent input and kernel as complex number
    vector<vector<Complex>> input_complex(n_fast, vector<Complex>(m_fast));
    vector<vector<Complex>> kernel_complex(n_fast, vector<Complex>(m_fast));
    for (int i = 0; i < n_fast; ++i) {
        for (int j = 0; j < m_fast; ++j) {
            input_complex[i][j] = input[i][j];
            kernel_complex[i][j] = kernel[i][j];
        }
    }

    // Compute FFT for input and kernel
    cout << "Computing FFT for input and kernel" << endl;
    gettimeofday(&start, 0);
    gettimeofday(&start2, 0);
    vector<vector<Complex>> input_fft = fft2d(input_complex, method);
    gettimeofday(&end2, 0);
    cout << "    FFT for input - " << (end2.tv_sec - start2.tv_sec) + (end2.tv_usec - start2.tv_usec) / 1e6 << "s" << endl;
    gettimeofday(&start2, 0);
    vector<vector<Complex>> kernel_fft = fft2d(kernel_complex, method);
    gettimeofday(&end2, 0);
    cout << "    FFT for kernel - " << (end2.tv_sec - start2.tv_sec) + (end2.tv_usec - start2.tv_usec) / 1e6 << "s" << endl;
    gettimeofday(&end, 0);
    cout << " > " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // Compute the convolution in frequency domain (multiplication)
    cout << "Computing the convolution in frequency domain";
    gettimeofday(&start, 0);
    vector<vector<Complex>> result(n_fast, vector<Complex>(m_fast));
    for (int i = 0; i < n_fast; ++i) {
        for (int j = 0; j < m_fast; ++j) {
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
    for (int i = 0; i < n_fast; ++i) {
        for (int j = 0; j < m_fast; ++j) {
            result_magnitude[i][j] = abs(result_complex[i][j]);
        }
    }
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // Crop the center of the result
    cout << "Cropping according to the given mode";
    gettimeofday(&start, 0);
    cropOutput(result_magnitude, n, m, n_kernel, m_kernel, n_fast, m_fast, mode);
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl << endl;

    return result_magnitude;
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
        py::arg("input"), py::arg("kernel"), py::arg("method")="mixed_radix", py::arg("mode")="full");
}
