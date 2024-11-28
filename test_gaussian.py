import fft
import numpy as np
import scipy.signal
import cv2

img = cv2.imread('assets/Lenna.png').astype(np.float64)
orig_h, orig_w, _ = img.shape
# img = cv2.resize(img, (4062, 4062))
h, w, _ = img.shape

gaussianKernel1d = cv2.getGaussianKernel(35, 10.0)
gaussianKernel2d = np.outer(gaussianKernel1d, gaussianKernel1d)

output_img = np.zeros((h, w, 3), dtype=np.float64)
output_img_scipy = np.zeros((h, w, 3), dtype=np.float64)
output_img_scipy_fft = np.zeros((h, w, 3), dtype=np.float64)

for channel in range(3):
    output_channel = fft.fftconvolve2d(img[:, :, channel], gaussianKernel2d)
    output_scipy = scipy.signal.convolve2d(img[:, :, channel], gaussianKernel2d, mode='same')
    output_scipy_fft = scipy.signal.fftconvolve(img[:, :, channel], gaussianKernel2d, mode='same')

    output_img[:, :, channel] = output_channel
    output_img_scipy[:, :, channel] = output_scipy
    output_img_scipy_fft[:, :, channel] = output_scipy_fft

cv2.imwrite('assets/output.png', np.clip(output_img, 0, 255).astype(np.uint8))
cv2.imwrite('assets/output_scipy.png', np.clip(output_img_scipy, 0, 255).astype(np.uint8))
cv2.imwrite('assets/output_scipy_fft.png', np.clip(output_img_scipy_fft, 0, 255).astype(np.uint8))

print("err (scipy conv): ", np.sum(np.abs(output_img - output_img_scipy)))
print("err (scipy fft): ", np.sum(np.abs(output_img - output_img_scipy_fft)))