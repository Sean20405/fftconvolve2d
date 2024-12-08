# ifndef FFT_IMPL_HPP
# define FFT_IMPL_HPP
# include <iostream>
# include <vector>
# include <complex>
# include <omp.h>
# include <sys/time.h>

extern "C" {
    #include "pocketfft/pocketfft.h"
};

typedef std::complex<double> Complex;
using namespace std;

extern vector<Complex> twiddle;

class CooleyTukeyFFT {
public:
    static vector<Complex> fft1d(vector<Complex> &input);
    static vector<Complex> ifft1d(vector<Complex> &input, bool root=true);
    static vector<vector<Complex>> fft2d(vector<vector<Complex>> &input);
    static vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input);
    static vector<int> fast_size;
};

class CooleyTukeyFFT_MP {
public:
    static vector<Complex> fft1d(vector<Complex> &input, int n_init);
    static vector<Complex> ifft1d(vector<Complex> &input,  int n_init, bool root=true);
    static vector<vector<Complex>> fft2d(vector<vector<Complex>> &input);
    static vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input);
    static vector<int> fast_size;
};

class MixedRadixFFT {
public:
    static vector<Complex> fft1d(vector<Complex> &input);
    static vector<Complex> ifft1d(vector<Complex> &input);
    static vector<vector<Complex>> fft2d(vector<vector<Complex>> &input);
    static vector<vector<Complex>> ifft2d(vector<vector<Complex>> &input);
    static vector<int> fast_size;
};

# endif // FFT_IMPL_HPP