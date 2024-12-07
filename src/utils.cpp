# include "utils.hpp"

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
pair<int, int> paddingInput(vector<vector<double>> &input, int n, int m, string method) {
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
    vector<vector<double>> padded_matrix(n_fast, vector<double>(m_fast, 0.0));
    for (int i=0; i<n_input; ++i) {
        for (int j=0; j<m_input; ++j) {
            padded_matrix[n_padding_size + i][m_padding_size + j] = input[i][j];
        }
    }
    input = padded_matrix;
    gettimeofday(&end, 0);
    cout << "        Padding rows - " << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6 << "s" << endl;
    cout << "        Original Size: " << n_input << "x" << m_input << " , Padding to size " << n_fast << "x" << m_fast << endl;

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
