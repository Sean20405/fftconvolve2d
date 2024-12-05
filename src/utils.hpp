# ifndef UTILS_HPP
# define UTILS_HPP
# include "fft_impl.hpp"
# include <iostream>
# include <vector>
# include <string>
# include <sys/time.h>

using namespace std;

void paddingKernel(vector<vector<double>> &kernel, int n, int m);
vector<vector<double>> roll2d(vector<vector<double>> &input, int shift_x, int shift_y);
pair<int, int> paddingInput(vector<vector<double>> &input, int n, int m, string method);
void cropOutput(vector<vector<double>> &input, int n, int m, int n_kernel, int m_kernel, int n_fast, int m_fast, string mode);

# endif // UTILS_HPP