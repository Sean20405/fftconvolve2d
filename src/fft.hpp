# include "fft_impl.hpp"
# include "utils.hpp"
# include <iostream>
# include <vector>
# include <complex>
# include <string>

extern "C" {
    #include "pocketfft/pocketfft.h"
};

typedef std::complex<double> Complex;
using namespace std;

vector<Complex> fft1d(vector<Complex> &input, string method="mixed_radix");
vector<Complex> ifft1d(vector<Complex> &input, string method="mixed_radix");
vector<vector<Complex>> fft2d(vector<vector<Complex>> &input, string method="mixed_radix");
vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input, string method="mixed_radix");

vector<vector<double>> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel, string method="mixed_radix", string mode="full");