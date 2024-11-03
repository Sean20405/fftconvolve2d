# include <iostream>
# include <vector>
# include <complex>
typedef std::complex<double> Complex;
using namespace std;

vector<Complex> fft1d(vector<Complex> &input);
vector<Complex> ifft1d(vector<Complex> &input, bool parent = true);
vector<vector<Complex>> fft2d(vector<vector<Complex>> &input);
vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input);

vector<vector<double>> fftconvolve2d(vector<vector<double>> &input, vector<vector<double>> &kernel);
vector<vector<double>> paddingKernel(vector<vector<double>> &kernel, int n, int m);