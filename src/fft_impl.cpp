# include "fft_impl.hpp"

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