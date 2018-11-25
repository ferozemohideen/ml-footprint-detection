# # from PIL import Image
# import numpy as np
# # import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
# #
# # #fig, axes = plt.subplots(3, 3, figsize=(9,24))
# im = plt.imread('images/unmarked/train_cnn/Female Archback 4.jpg')
#
# from scipy import signal
# from scipy import misc
# lena = misc.ascent() - misc.ascent().mean()
# template = np.copy(lena[235:295, 310:370]) # right eye
# template -= template.mean()
# lena = lena + np.random.randn(*lena.shape) * 50 # add noise
# lena=im[:,:,2]
# im2 = plt.imread('images/unmarked/masks/Female Archback 4.jpg')
# template=im2[:,:,2]
# corr = signal.correlate2d(lena, template, boundary='symm', mode='same')
# y, x = np.unravel_index(np.argmax(corr), corr.shape) # find the match
#
# import matplotlib.pyplot as plt
# fig, (ax_orig, ax_template, ax_corr) = plt.subplots(1, 3)
# ax_orig.imshow(lena, cmap='gray')
# ax_orig.set_title('Original')
# ax_orig.set_axis_off()
# ax_template.imshow(template, cmap='gray')
# ax_template.set_title('Template')
# ax_template.set_axis_off()
# ax_corr.imshow(corr, cmap='gray')
# ax_corr.set_title('Cross-correlation')
# ax_corr.set_axis_off()
# ax_orig.plot(x, y, 'ro')
# plt.show()


# Python program to illustrate
# template matching
import cv2
import numpy as np

# Read the main image
img_rgb = cv2.imread('images/unmarked/train_cnn/Female Archback 4.jpg')
# img_rgb = cv2.resize(img_rgb, (0,0), fx=0.8, fy=0.8)
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread('images/unmarked/masks/Female Archback 4.jpg', 0)
template = cv2.resize(template, (0,0), fx=0.8, fy=0.8)

# Store width and heigth of template in w and h
w, h = template.shape[::-1]
template = cv2.getRotationMatrix2D((w/2,h/2),90,1)
template = cv2.warpAffine(img_gray,template,(w,h))
# Perform match operations.
#res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#
# # Specify a threshold
# threshold = 0.8
#
# # Store the coordinates of matched area in a numpy array
# loc = np.where(res >= threshold)
#
# # Draw a rectangle around the matched region.
# for pt in zip(*loc[::-1]):
#      cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
#
# # Show the final image with the matched area.
# plt.imshow('Detected', img_rgb)

#corr = signal.correlate2d(img_gray, template, boundary='symm', mode='same')

image = cv2.imread('images/unmarked/train_cnn/Female Archback 4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 20, 100)

plt.imshow(edged)
plt.show()