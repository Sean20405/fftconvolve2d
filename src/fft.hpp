# include "fft_impl.hpp"
# include "utils.hpp"
# include <iostream>
# include <vector>
# include <complex>
# include <string>
# include <pybind11/pybind11.h>
# include <pybind11/stl.h>
# include <pybind11/operators.h>
# include <pybind11/complex.h>
# include <pybind11/numpy.h>

extern "C" {
    #include "pocketfft/pocketfft.h"
};

typedef std::complex<double> Complex;
namespace py = pybind11;

vector<Complex> fft1d(vector<Complex> &input, string method="mixed_radix");
vector<Complex> ifft1d(vector<Complex> &input, string method="mixed_radix");
vector<vector<Complex>> fft2d(vector<vector<Complex>> &input, string method="mixed_radix");
vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input, string method="mixed_radix");

py::array_t<double> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel, string method="mixed_radix", string mode="full");