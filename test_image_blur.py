from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

fig, axes = plt.subplots(1, 2, figsize=(12,5))
def pool(image, fn, kernel=5, stride=2):
    h_prev, w_prev, n = image.shape

    h = int((h_prev - kernel) / stride) + 1
    w = int((w_prev - kernel) / stride) + 1

    downsampled = np.zeros((h, w, n))

    for i in range(n):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + kernel <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + kernel <= w_prev:
                # choose the maximum value within the window at each step and store it to   the output matrix

                downsampled[out_y, out_x, i] = fn(image[curr_y:curr_y + kernel,
                                                  curr_x:curr_x + kernel, i])

                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return downsampled


def scan(image, filt, stride=1):
    kernel = filt.shape[0]
    h_prev, w_prev, n = image.shape

    h = int((h_prev - kernel) / stride) + 1
    w = int((w_prev - kernel) / stride) + 1
    downsampled = np.zeros((h, w, n))

    for i in range(n):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + kernel <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + kernel <= w_prev:
                # choose the maximum value within the window at each step and store it to   the output matrix

                downsampled[out_y, out_x, i] = np.sum(filt *
                                                      image[curr_y:curr_y + kernel,
                                                      curr_x:curr_x + kernel, i])

                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return downsampled

img = cv2.imread('images/unmarked/test_cnn/Female Archback 1.jpg')
blur = cv2.bilateralFilter(img,40,100,100)


# print(blur2.shape)
# cv2.imshow('blur', blur2)
# cv2.imshow('original', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#plt.imsave('images/unmarked/filter/Female Archback 4.jpg', blur)
def detectEdges(arg, size=5):
    img = pool(arg[:, :, :2], np.mean, kernel=50)
    hor = np.zeros((size, size))
    hor[size // 2 - 1, :] = -10
    hor[size // 2, :] = 10

    vert = np.zeros((size, size))
    vert[:, size // 2 - 1] = -10
    vert[:, size // 2] = 10

    img1 = scan(img[:, :, :2], hor)
    img2 = scan(img[:, :, :2], vert)

    img1 = np.abs(img1)
    img2 = np.abs(img2)

    img = img2 + img1
    return np.sum(img, axis=2)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
# blur2 = auto_canny(blur)
# blur2 = auto_canny(blur)
sns.heatmap(detectEdges(blur), axes=axes[1],cbar=None)
axes[0].imshow(img)
plt.tight_layout()
plt.axis("off")
axes[0].axis("off")
plt.show()

# plt.show()