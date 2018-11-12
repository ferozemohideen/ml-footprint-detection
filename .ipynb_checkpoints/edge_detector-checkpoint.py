import base64
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''with open('../Desktop/base64.txt', 'rb') as imgfile:
    img = base64.decodestring(imgfile.read())

with open('../Desktop/dot-matt.jpg', 'wb') as f:
    f.write(img)'''

im = Image.open('images/Female Archback 1.jpg')

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(im)
img = np.array(im.getdata()).reshape((2376,3990,3))
# im = Image.open('../Desktop/dot.jpg')
# img = np.array(im.getdata()).reshape((478,587,3))


#sns.heatmap(img[:,:,0])

def pool(image, fn, kernel=10, stride=15):

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

def scan(image, filt, stride=2):

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




#print(img.shape)
img = pool(img[:,:,:2], np.max)

size = 5
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

#img = pool(img, np.max, kernel=10, stride=1)
#img = pool(img, np.min, kernel=5, stride=1)
'''
img = img > 1000

true_idx = np.where(img[:,:,0])
x, y = true_idx[0], true_idx[1]

print((min(x), min(y)),
      (min(x), max(y)),
      (max(x), min(y)),
      (max(x), max(y)))
'''


img = np.sum(img, axis=2)

#dilation and erosion
#vect = img.copy().flatten().sort()
#for i in range(len(vect)-1):
#   vect[i] + vect[i+1]


sns.heatmap(img, ax= axes[1])
#sns.heatmap(img1[:,:,0])


plt.show()



'''im = Image.open('../Desktop/dot.jpg', 'r')
pix = im.load()
print(im.size)'''