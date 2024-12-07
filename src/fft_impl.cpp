# include "fft_impl.hpp"

vector<int> CooleyTukeyFFT::fast_size = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

vector<Complex> CooleyTukeyFFT::fft1d(vector<Complex> &input) {
    int n = input.size();
    if (n == 1) {
        return input;
    }

    // Divide into even and odd function
    vector<Complex> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; i++) {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }
    
    // Recursively compute FFT
    vector<Complex> even_result = fft1d(even);
    vector<Complex> odd_result = fft1d(odd);

    // Merge
    vector<Complex> result(n);
    double angle = -2 * M_PI / n;
    Complex omega(1, 0), omega_n(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; i++) {
        result[i] = even_result[i] + omega * odd_result[i];
        result[i + n / 2] = even_result[i] - omega * odd_result[i];
        omega *= omega_n;
    }

    return result;
}

vector<Complex> CooleyTukeyFFT::ifft1d(vector<Complex> &input, bool root) {
    int n = input.size();
    if (n == 1) {
        return input;
    }

    // Divide into even and odd function
    vector<Complex> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; i++) {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }

    // Recursively compute FFT
    vector<Complex> even_result = ifft1d(even, false);
    vector<Complex> odd_result = ifft1d(odd, false);

    // Merge
    vector<Complex> result(n);
    double angle = 2 * M_PI / n;
    Complex omega(1, 0), omega_n(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; i++) {
        result[i] = even_result[i] + omega * odd_result[i];
        result[i + n / 2] = even_result[i] - omega * odd_result[i];
        omega *= omega_n;
    }

    if (root) {
        for (int i = 0; i < n; i++) {
            result[i] /= n;
        }
    }

    return result;
}

// 2D FFT
vector<vector<Complex>> CooleyTukeyFFT::fft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();

    vector<vector<Complex>> result(n, vector<Complex>(m));

    // FFT for each row  TODO: bottleneck
    for (int i = 0; i < n; i++) {
        result[i] = fft1d(input[i]);
    }

    // FFT for each column  TODO: bottleneck
    for (int i = 0; i < m; i++) {
        vector<Complex> column(n);
        for (int j = 0; j < n; j++) {
            column[j] = result[j][i];
        }
        column = fft1d(column);
        for (int j = 0; j < n; j++) {
            result[j][i] = column[j];
        }
    }

    return result;
}

vector<vector<Complex>> CooleyTukeyFFT::ifft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();
    vector<vector<Complex>> result(n, vector<Complex>(m));

    // FFT for each row
    for (int i = 0; i < n; i++) {
        result[i] = ifft1d(input[i]);
    }

    // FFT for each column
    for (int i = 0; i < m; i++) {
        vector<Complex> column(n);
        for (int j = 0; j < n; j++) {
            column[j] = result[j][i];
        }
        column = ifft1d(column);
        for (int j = 0; j < n; j++) {
            result[j][i] = column[j];
        }
    }

    return result;
}

vector<int> CooleyTukeyFFT_MP::fast_size = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

vector<Complex> CooleyTukeyFFT_MP::fft1d(vector<Complex> &input) {
    int n = input.size();
    if (n == 1) {
        return input;
    }

    // Divide into even and odd function
    vector<Complex> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; i++) {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }
    
    // Recursively compute FFT
    vector<Complex> even_result = fft1d(even);
    vector<Complex> odd_result = fft1d(odd);

    // Merge
    vector<Complex> result(n);
    double angle = -2 * M_PI / n;
    Complex omega(1, 0), omega_n(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; i++) {
        result[i] = even_result[i] + omega * odd_result[i];
        result[i + n / 2] = even_result[i] - omega * odd_result[i];
        omega *= omega_n;
    }

    return result;
}

vector<Complex> CooleyTukeyFFT_MP::ifft1d(vector<Complex> &input, bool root) {
    int n = input.size();
    if (n == 1) {
        return input;
    }

    // Divide into even and odd function
    vector<Complex> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; i++) {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }

    // Recursively compute FFT
    vector<Complex> even_result = ifft1d(even, false);
    vector<Complex> odd_result = ifft1d(odd, false);

    // Merge
    vector<Complex> result(n);
    double angle = 2 * M_PI / n;
    Complex omega(1, 0), omega_n(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; i++) {
        result[i] = even_result[i] + omega * odd_result[i];
        result[i + n / 2] = even_result[i] - omega * odd_result[i];
        omega *= omega_n;
    }

    if (root) {
        for (int i = 0; i < n; i++) {
            result[i] /= n;
        }
    }

    return result;
}

// 2D FFT
vector<vector<Complex>> CooleyTukeyFFT_MP::fft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();

    vector<vector<Complex>> result(n, vector<Complex>(m));

    // FFT for each row  TODO: bottleneck
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = fft1d(input[i]);
    }

    // FFT for each column  TODO: bottleneck
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        vector<Complex> column(n);
        for (int j = 0; j < n; j++) {
            column[j] = result[j][i];
        }
        column = fft1d(column);
        for (int j = 0; j < n; j++) {
            result[j][i] = column[j];
        }
    }

    return result;
}

vector<vector<Complex>> CooleyTukeyFFT_MP::ifft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();
    vector<vector<Complex>> result(n, vector<Complex>(m));

    // FFT for each row
    #pragma omp simd    
    for (int i = 0; i < n; i++) {
        result[i] = ifft1d(input[i]);
    }

    // FFT for each column
    #pragma omp simd    
    for (int i = 0; i < m; i++) {
        vector<Complex> column(n);
        for (int j = 0; j < n; j++) {
            column[j] = result[j][i];
        }
        column = ifft1d(column);
        for (int j = 0; j < n; j++) {
            result[j][i] = column[j];
        }
    }

    return result;
}

vector<int> MixedRadixFFT::fast_size = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 81, 84, 90, 96, 98, 100, 105, 108, 112, 120, 125, 126, 128, 135, 140, 144, 147, 150, 160, 162, 168, 175, 180, 189, 192, 196, 200, 210, 216, 224, 225, 240, 243, 245, 250, 252, 256, 270, 280, 288, 294, 300, 315, 320, 324, 336, 343, 350, 360, 375, 378, 384, 392, 400, 405, 420, 432, 441, 448, 450, 480, 486, 490, 500, 504, 512, 525, 540, 560, 567, 576, 588, 600, 625, 630, 640, 648, 672, 675, 686, 700, 720, 729, 735, 750, 756, 768, 784, 800, 810, 840, 864, 875, 882, 896, 900, 945, 960, 972, 980, 1000, 1008, 1024, 1029, 1050, 1080, 1120, 1125, 1134, 1152, 1176, 1200, 1215, 1225, 1250, 1260, 1280, 1296, 1323, 1344, 1350, 1372, 1400, 1440, 1458, 1470, 1500, 1512, 1536, 1568, 1575, 1600, 1620, 1680, 1701, 1715, 1728, 1750, 1764, 1792, 1800, 1875, 1890, 1920, 1944, 1960, 2000, 2016, 2025, 2048, 2058, 2100, 2160, 2187, 2205, 2240, 2250, 2268, 2304, 2352, 2400, 2401, 2430, 2450, 2500, 2520, 2560, 2592, 2625, 2646, 2688, 2700, 2744, 2800, 2835, 2880, 2916, 2940, 3000, 3024, 3072, 3087, 3125, 3136, 3150, 3200, 3240, 3360, 3375, 3402, 3430, 3456, 3500, 3528, 3584, 3600, 3645, 3675, 3750, 3780, 3840, 3888, 3920, 3969, 4000, 4032, 4050, 4096, 4116, 4200, 4320, 4374, 4375, 4410, 4480, 4500, 4536, 4608, 4704, 4725, 4800, 4802, 4860, 4900, 5000, 5040, 5103, 5120, 5145, 5184, 5250, 5292, 5376, 5400, 5488, 5600, 5625, 5670, 5760, 5832, 5880, 6000, 6048, 6075, 6125, 6144, 6174, 6250, 6272, 6300, 6400, 6480, 6561, 6615, 6720, 6750, 6804, 6860, 6912, 7000, 7056, 7168, 7200, 7203, 7290, 7350, 7500, 7560, 7680, 7776, 7840, 7875, 7938, 8000, 8064, 8100, 8192, 8232, 8400, 8505, 8575, 8640, 8748, 8750, 8820, 8960, 9000, 9072, 9216, 9261, 9375, 9408, 9450, 9600, 9604, 9720, 9800, 10000, 10080, 10125, 10206, 10240, 10290, 10368, 10500, 10584, 10752, 10800, 10935, 10976, 11025, 11200, 11250, 11340, 11520, 11664, 11760, 11907, 12000, 12005, 12096, 12150, 12250, 12288, 12348, 12500, 12544, 12600, 12800, 12960, 13122, 13125, 13230, 13440, 13500, 13608, 13720, 13824, 14000, 14112, 14175, 14336, 14400, 14406, 14580, 14700, 15000, 15120, 15309, 15360, 15435, 15552, 15625, 15680, 15750, 15876, 16000, 16128, 16200, 16384, 16464, 16800, 16807, 16875, 17010, 17150, 17280, 17496, 17500, 17640, 17920, 18000, 18144, 18225, 18375, 18432, 18522, 18750, 18816, 18900, 19200, 19208, 19440, 19600, 19683, 19845, 20000};

vector<Complex> MixedRadixFFT::fft1d(vector<Complex> &input) {
cfft_plan plan = make_cfft_plan(input.size());

    // Convert input to double array that can be used in pocketfft
    double *data = new double[2 * input.size()];
    for (size_t i = 0; i < input.size(); i++) {
        data[2 * i] = input[i].real();
        data[2 * i + 1] = input[i].imag();
    }

    // Compute FFT
    cfft_forward(plan, data, 1.0);

    // Convert the result to Complex array
    vector<Complex> result(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        result[i] = Complex(data[2 * i], data[2 * i + 1]);
    }

    // Free memory
    delete[] data;
    destroy_cfft_plan(plan);

    return result;
}

vector<Complex> MixedRadixFFT::ifft1d(vector<Complex> &input) {
    cfft_plan plan = make_cfft_plan(input.size());

    // Convert input to double array that can be used in pocketfft
    double *data = new double[2 * input.size()];
    for (size_t i = 0; i < input.size(); i++) {
        data[2 * i] = input[i].real();
        data[2 * i + 1] = input[i].imag();
    }

    // Compute inverse FFT
    cfft_backward(plan, data, 1.0/input.size());

    // Convert the result to Complex array
    vector<Complex> result(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        result[i] = Complex(data[2 * i], data[2 * i + 1]);
    }

    // Free memory
    delete[] data;
    destroy_cfft_plan(plan);

    return result;
}

vector<vector<Complex>> MixedRadixFFT::fft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();
    struct timeval start, end;

    vector<vector<Complex>> result(n, vector<Complex>(m));

    // FFT for each row  TODO: bottleneck
    cout << "    FFT for each row";
    gettimeofday(&start, 0);
    for (int i = 0; i < n; i++) {
        result[i] = fft1d(input[i]);
    }
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // FFT for each column  TODO: bottleneck
    cout << "    FFT for each column";
    vector<Complex> column(n);
    gettimeofday(&start, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            column[j] = result[j][i];
        }
        column = fft1d(column);
        for (int j = 0; j < n; j++) {
            result[j][i] = column[j];
        }
    }
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    return result;
}

vector<vector<Complex>> MixedRadixFFT::ifft2d(vector<vector<Complex>> &input) {
    int n = input.size();
    int m = input[0].size();
    vector<vector<Complex>> result(n, vector<Complex>(m));

    // FFT for each row
    struct timeval start, end;
    cout << "    IFFT for each row";
    gettimeofday(&start, 0);
    for (int i = 0; i < n; i++) {
        result[i] = ifft1d(input[i]);
    }
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    // FFT for each column
    cout << "    IFFT for each column";
    vector<Complex> column(n);
    gettimeofday(&start, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            column[j] = result[j][i];
        }
        column = ifft1d(column);
        for (int j = 0; j < n; j++) {
            result[j][i] = column[j];
        }
    }
    gettimeofday(&end, 0);
    cout << " - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;

    return result;
}