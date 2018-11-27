# # from PIL import Image
# import numpy as np
# # import seaborn as sns
import matplotlib.pyplot as plt
# Python program to illustrate
# template matching
import cv2
import numpy as np

# Read the main image
img_rgb = cv2.imread('images/unmarked/train_cnn/Female Majanna 1.jpg')
# img_rgb = cv2.resize(img_rgb, (0,0), fx=0.8, fy=0.8)
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread('images/unmarked/masks/Female Archback 4.jpg', 0)
#template = cv2.resize(template, (0,0), fx=0.8, fy=0.8)

# Store width and heigth of template in w and h
w, h = template.shape

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=255)
  return result

def fit(template, img_gray):
    res = -100000
    maxImage = []
    angle = 0
    for deg in np.arange(-15,15):
        for size in np.arange(0.8, 1, 0.01):
            resized_template = cv2.resize(template, (0, 0), fx=size, fy=size)
            bordersize_bot = img_gray.shape[0] - resized_template.shape[0]
            bordersize_right = img_gray.shape[1] - resized_template.shape[1]
            resized_template = cv2.copyMakeBorder(resized_template, top=0, left=0, bottom=bordersize_bot,
                                                  right=bordersize_right, borderType=cv2.BORDER_CONSTANT,
                                                  value=[255, 255, 255])
            img = rotateImage(resized_template, deg)
            temp = cv2.matchTemplate(img, img_gray, cv2.TM_CCOEFF_NORMED)
            if temp > res:
                maxImage = img
                res = temp
                angle = deg
    return (maxImage, res, angle)

def resize_template(template, img_gray):
    # bordersize_bot = img_gray.shape[0]-template.shape[0]
    # bordersize_right = img_gray.shape[1]-template.shape[1]
    # border=cv2.copyMakeBorder(template, top=0, left=0, bottom=bordersize_bot,
    #                           right=bordersize_right, borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
    newsize = tuple(np.array(img_gray.shape))
    resized_image = cv2.resize(template, newsize[::-1])
    border = resized_image

    return border




# (maxImage, res, angle) = fit(resize_template(template, img_gray), img_gray)
# temp = cv2.addWeighted(img_gray, 1, maxImage, 0.1, 1)
#
# fig, axes = plt.subplots(1, 3, figsize=(12, 3))
# axes[0].imshow(img_gray)
#
# axes[1].imshow(maxImage)
#
# axes[2].imshow(temp, cmap='Greys')
# plt.show()

# M = np.float32([[1,0,200],[0,1,0]])
# dst = cv2.warpAffine(template,M,(h,w), borderValue=255)
# plt.imshow(dst)
# plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/unmarked/train_cnn/Female Archback 4.jpg',0)
img2 = img.copy()
template = cv2.imread('images/unmarked/test.jpg',0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCORR']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Footprint'), plt.xticks([]), plt.yticks([])
    plt.suptitle("Method Used: " + meth)

    plt.show()