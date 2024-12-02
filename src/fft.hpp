# include "fft_impl.hpp"
# include <iostream>
# include <vector>
# include <complex>

extern "C" {
    #include "pocketfft/pocketfft.h"
};

typedef std::complex<double> Complex;
using namespace std;

vector<Complex> fft1d(vector<Complex> &input, string method="mixed_radix");
vector<Complex> ifft1d(vector<Complex> &input, string method="mixed_radix");
vector<vector<Complex>> fft2d(vector<vector<Complex>> &input, string method="mixed_radix");
vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input, string method="mixed_radix");

vector<vector<double>> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel, string method="mixed_radix");
void paddingKernel(vector<vector<double>> &kernel, int n, int m);
vector<vector<double>> roll2d(vector<vector<double>> &input, int shift_x, int shift_y);
pair<int, int> paddingInput(vector<vector<double>> &input, int n, int m, string &method);
