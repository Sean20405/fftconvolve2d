# include <iostream>
# include <vector>
# include <complex>
# include <sys/time.h>

extern "C" {
    #include "pocketfft/pocketfft.h"
};

typedef std::complex<double> Complex;
using namespace std;

class CooleyTukeyFFT {
public:
    static vector<Complex> fft1d(vector<Complex> &input);
    static vector<Complex> ifft1d(vector<Complex> &input, bool root=true);
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