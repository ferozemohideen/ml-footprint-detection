
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

fig, axes = plt.subplots(nrows=2, ncols=2)
img = cv2.imread('images/Female Archback 1.jpg',0)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow,ccol = round(rows/2) , round(cols/2)

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

axes[0,0].imshow(img, cmap='gray')

plt.imsave('test.jpg', img_back)
im = Image.open('test.jpg')

size = im.getdata().size
axes[0,1].imshow(im, cmap='gray')
img = np.array(im.getdata()).reshape((size[1],size[0],3))

def pool(image, fn, kernel=5, stride=2):

    h_prev, w_prev, n = image.shape

    h = int((h_prev - kernel)/stride)+1
    w = int((w_prev - kernel)/stride)+1

    downsampled = np.zeros((h, w, n))

    for i in range(n):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + kernel <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + kernel <= w_prev:
                # choose the maximum value within the window at each step and store it to   the output matrix

                downsampled[out_y, out_x, i] = fn(image[curr_y:curr_y+kernel,
                                                   curr_x:curr_x+kernel, i])

                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return downsampled

def scan(image, filt, stride=1):

    kernel=filt.shape[0]
    h_prev, w_prev, n = image.shape

    h = int((h_prev - kernel)/stride)+1
    w = int((w_prev - kernel)/stride)+1
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
                                                      image[curr_y:curr_y+kernel,
                                                      curr_x:curr_x+kernel, i])

                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return downsampled



size = 5
speckle = np.zeros((1, size*size))
for i in range(size*size):
    speckle[0, i] = 1 if i%2 else -1
speckle = speckle.reshape((size, size))

bigspeckle = np.array([[ 1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1.],
                        [ 1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1.],
                       [1., 1., 1., -1., -1., -1., 1., 1., 1.],
                        [-1., -1., -1.,  1.,  1.,  1., -1., -1., -1.],
                       [-1., -1., -1., 1., 1., 1., -1., -1., -1.],
                       [-1., -1., -1., 1., 1., 1., -1., -1., -1.],
                         [ 1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1.],
                       [1., 1., 1., -1., -1., -1., 1., 1., 1.],
                       [1., 1., 1., -1., -1., -1., 1., 1., 1.],])
#print(bigspeckle)
img = pool(img[:,:,:2], np.mean, kernel=50)
# img = scan(img[:,:,:2], speckle)
# img = pool(img[:,:,:2], np.mean, kernel=20)
# img = scan(img[:,:,:2], bigspeckle)
# img = scan(img[:,:,:2], bigspeckle)
# #img = scan(img[:,:,:2], speckle)
# #img = pool(img[:,:,:2], np.min)
# #print(img.shape)
# #img = pool(img[:,:,:], np.max)
# #print(img.shape)
#
#
hor = np.zeros((size, size))
hor[size//2-1,:] = -10
hor[size//2,:] = 10

vert = np.zeros((size, size))
vert[:,size//2-1] = -10
vert[:,size//2] = 10

img1 = scan(img[:,:,:2], hor)
img2 = scan(img[:,:,:2], vert)


img1 = np.abs(img1)
img2 = np.abs(img2)

img = img2 + img1




img = np.sum(img, axis=2)


#dilation and erosion
#vect = img.copy().flatten().sort()
#for i in range(len(vect)-1):
#   vect[i] + vect[i+1]

sns.heatmap(img, ax=axes[1,0])

g = img.shape
img = img.reshape((1,img.size))
arr = []
for i in range(img.size):
    if img.item(i) > 150:
        arr.append(1)
    else:
        arr.append(-1)
arr = np.array(arr)
img = arr.reshape(g)
sns.heatmap(img, ax=axes[1,1])

plt.show()


# tfcm