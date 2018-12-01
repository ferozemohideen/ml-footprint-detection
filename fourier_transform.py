import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

img = cv2.imread('images/unmarked/test_cnn/Female Archback 1.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

fig,axes = plt.subplots(1,2, figsize=(10,5))

# plt.subplot(321),plt.imshow(img, cmap='gray')
# plt.title('Full Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(322),sns.heatmap(magnitude_spectrum)
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#
img = cv2.imread('images/marked/rough_sand.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

axes[0].imshow(img, cmap='gray')
axes[0].axis('off')
sns.heatmap(magnitude_spectrum, ax=axes[1],cbar=None)
axes[1].axis('off')
plt.tight_layout()
plt.show()

# img = cv2.imread('images/marked/smooth_sand.jpg',0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
#
# plt.subplot(325),plt.imshow(img, cmap='gray')
# plt.title('Smooth Sand'), plt.xticks([]), plt.yticks([])
# plt.subplot(326),sns.heatmap(magnitude_spectrum)
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])




