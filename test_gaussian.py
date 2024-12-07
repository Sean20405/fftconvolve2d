import fft
import numpy as np
import scipy.signal
import cv2

img = cv2.imread('assets/Lenna.png').astype(np.float64)
orig_h, orig_w, _ = img.shape
img = cv2.resize(img, (4096, 4096))
h, w, _ = img.shape

gaussianKernel1d = cv2.getGaussianKernel(35, 10.0)
gaussianKernel2d = np.outer(gaussianKernel1d, gaussianKernel1d)

mode = 'full'

if mode == 'full':
    output_shape = (h + 34, w + 34, 3)
elif mode == 'same':
    output_shape = (h, w, 3)
elif mode == 'valid':
    output_shape = (h - 34, w - 34, 3)

output_img = np.zeros(output_shape, dtype=np.float64)
output_img_scipy = np.zeros(output_shape, dtype=np.float64)
output_img_scipy_fft = np.zeros(output_shape, dtype=np.float64)

mode = 'full'

for channel in range(3):
    output_channel = fft.fftconvolve2d(img[:, :, channel], gaussianKernel2d, mode=mode, method='cooley_tukey')
    output_scipy = scipy.signal.convolve2d(img[:, :, channel], gaussianKernel2d, mode=mode)
    output_scipy_fft = scipy.signal.fftconvolve(img[:, :, channel], gaussianKernel2d, mode=mode)

    output_img[:, :, channel] = output_channel
    output_img_scipy[:, :, channel] = output_scipy
    output_img_scipy_fft[:, :, channel] = output_scipy_fft

cv2.imwrite('assets/output.png', np.clip(output_img, 0, 255).astype(np.uint8))
cv2.imwrite('assets/output_scipy.png', np.clip(output_img_scipy, 0, 255).astype(np.uint8))
cv2.imwrite('assets/output_scipy_fft.png', np.clip(output_img_scipy_fft, 0, 255).astype(np.uint8))

print("err (scipy conv): ", np.sum(np.abs(output_img - output_img_scipy)))
print("err (scipy fft): ", np.sum(np.abs(output_img - output_img_scipy_fft)))