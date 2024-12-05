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

vector<vector<double>> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel, string method, string mode) {
    struct timeval start, end, start2, end2;

    int n = input.size(), m = input[0].size();
    int n_kernel = kernel.size(), m_kernel = kernel[0].size();
    int n_fast, m_fast;

    // Preprocess the image and kernel
    cout << "Preprocessing the image and kernel" << endl;
    cout << "    Padding to faster size";
    gettimeofday(&start2, 0);
    std::tie(n_fast, m_fast) = paddingInput(input, n + n_kernel, m + m_kernel, method);
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
    cout << "Cropping according to the given mode";
    gettimeofday(&start, 0);
    cropOutput(result_magnitude, n, m, n_kernel, m_kernel, n_fast, m_fast, mode);
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl << endl;

    return result_magnitude;
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


/* Pad zeros around the input to the smallest value in fast_size that is greater than or equal to the input size
   n, m: the size considering the padding of kernel, but not fast_size */
pair<int, int> paddingInput(vector<vector<double>> &input, int n, int m, string &method) {
    struct timeval start, end;
    int n_input = input.size(), m_input = input[0].size();
    int n_fast, m_fast;
    int n_padding_size, m_padding_size;

    // Find the smallest value in fast_size that is greater than or equal to n and m
    gettimeofday(&start, 0);
    if (method == "mixed_radix") {
        n_fast = *lower_bound(MixedRadixFFT::fast_size.begin(), MixedRadixFFT::fast_size.end(), n);
        m_fast = *lower_bound(MixedRadixFFT::fast_size.begin(), MixedRadixFFT::fast_size.end(), m);
    }
    else if (method == "cooley_tukey") {
        n_fast = *lower_bound(CooleyTukeyFFT::fast_size.begin(), CooleyTukeyFFT::fast_size.end(), n);
        m_fast = *lower_bound(CooleyTukeyFFT::fast_size.begin(), CooleyTukeyFFT::fast_size.end(), m);
    }
    else throw invalid_argument("Invalid method");  // This may be checked before calling this function
    gettimeofday(&end, 0);
    cout << endl << "        Choose method - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    n_padding_size = (n_fast - n_input) / 2, m_padding_size = (m_fast - m_input) / 2;

    // Padding rows  TODO: bottleneck
    gettimeofday(&start, 0);
    for (int i=0; i<n_input; i++) {
        input[i].resize(m_fast);
        for (int j=0; j<m_padding_size; j++) {
            input[i].insert(input[i].begin(), 0);
            input[i].push_back(0);
        }
    }
    gettimeofday(&end, 0);
    cout << "        Padding rows - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // Padding columns
    gettimeofday(&start, 0);
    input.resize(n_fast, vector<double>(m_fast));
    for (int i=0; i<n_padding_size; i++) {
        vector<double> padding(m_fast, 0);
        input.insert(input.begin(), padding);
        input.push_back(padding);
    }
    gettimeofday(&end, 0);
    cout << "        Padding columns - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;
    cout << "        Padding to size " << n_fast << "x" << m_fast << endl;

    return make_pair(n_fast, m_fast);
}

/* Recover from padding to fast size. Also crop input according to given mode */
void cropOutput(vector<vector<double>> &input, int n, int m, int n_kernel, int m_kernel, int n_fast, int m_fast, string mode) {
    int n_kernel_padding_thk = n_kernel / 2, m_kernel_padding_thk = m_kernel / 2;  // Padding thickness about kernel
    int n_fast_padding_thk = (n_fast - (n + n_kernel_padding_thk * 2)) / 2, m_fast_padding_thk = (m_fast - (m + m_kernel_padding_thk * 2)) / 2;  // Padding thickness for fast size
    
    // Recover from padding to fast size
    for (int i = 0; i < n + n_kernel_padding_thk * 2; i++) {
        for (int j = 0; j < m + m_kernel_padding_thk * 2; j++) {
            input[i][j] = input[i + n_fast_padding_thk][j + m_fast_padding_thk];
        }
    }

    // Crop image
    if (mode == "full") {
        input.resize(n + n_kernel_padding_thk * 2);
        for (int i = 0; i < n + n_kernel_padding_thk * 2; i++) {
            input[i].resize(m + m_kernel_padding_thk * 2);
        }
    }
    else if (mode == "same") {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                input[i][j] = input[i + n_kernel_padding_thk][j + m_kernel_padding_thk];
            }
        }

        input.resize(n);
        for (int i = 0; i < n; i++) {
            input[i].resize(m);
        }
    }
    else if (mode == "valid") {
        for (int i = 0; i < n - n_kernel_padding_thk * 2; i++) {
            for (int j = 0; j < m - m_kernel_padding_thk * 2; j++) {
                input[i][j] = input[i + 2 * n_kernel_padding_thk][j + 2 * m_kernel_padding_thk];
            }
        }

        input.resize(n - n_kernel_padding_thk * 2);
        for (int i = 0; i < n - n_kernel_padding_thk * 2; i++) {
            input[i].resize(m - m_kernel_padding_thk * 2);
        }
    }
    else throw invalid_argument("Invalid mode");

    return;
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
    m.def("paddingInput", &paddingInput, "Padding zero at the surrounding of the input until the size of input is n x m.");
}
