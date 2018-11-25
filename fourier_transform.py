import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

img = cv2.imread('images/Female Archback 1.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))


plt.subplot(321),plt.imshow(img, cmap='gray')
plt.title('Full Image'), plt.xticks([]), plt.yticks([])
plt.subplot(322),sns.heatmap(magnitude_spectrum)
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#
img = cv2.imread('images/rough_sand.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(323),plt.imshow(img, cmap='gray')
plt.title('Rough Sand'), plt.xticks([]), plt.yticks([])
plt.subplot(324),sns.heatmap(magnitude_spectrum)
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.imread('images/smooth_sand.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(325),plt.imshow(img, cmap='gray')
plt.title('Smooth Sand'), plt.xticks([]), plt.yticks([])
plt.subplot(326),sns.heatmap(magnitude_spectrum)
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# plt.tight_layout()
plt.show()

