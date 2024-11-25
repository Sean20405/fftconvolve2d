# include <iostream>
# include <vector>
# include <complex>

extern "C" {
    #include "pocketfft/pocketfft.h"
};

typedef std::complex<double> Complex;
using namespace std;

vector<Complex> fft1d(vector<Complex> &input);
vector<Complex> ifft1d(vector<Complex> &input);
vector<vector<Complex>> fft2d(vector<vector<Complex>> &input);
vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input);

vector<vector<double>> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel);
void paddingKernel(vector<vector<double>> &kernel, int n, int m);
vector<vector<double>> roll2d(vector<vector<double>> &input, int shift_x, int shift_y);
void paddingInput(vector<vector<double>> &input, int n, int m);