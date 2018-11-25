from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def pool(mask, image, fn, kernel=5, stride=2):
    h_prev, w_prev = mask.shape

    labels = []
    values = []


    curr_y = out_y = 0
    # slide the max pooling window vertically across the image
    while curr_y + kernel <= h_prev:
        curr_x = out_x = 0
        # slide the max pooling window horizontally across the image
        while curr_x + kernel <= w_prev:
            # choose the maximum value within the window at each step and store it to   the output matrix

            if fn(mask[curr_y:curr_y + kernel, curr_x:curr_x + kernel]):
                labels.append(1)
            else:
                labels.append(0)

            values.append(np.ravel(image[curr_y:curr_y + kernel, curr_x:curr_x + kernel, :]))

            curr_x += stride
            out_x += 1
        curr_y += stride
        out_y += 1

    return (np.array(labels), np.array(values))

def edge(img):
    return np.sum(img) >= 0
# im = Image.open('images/unmarked/masks/Female Archback 4.jpg')
# size = im.getdata().size
# img = np.array(im.getdata()).reshape((size[1],size[0],3))
# img = img[:,:,0]
# img[img>0] = 1
# img[img<=0] = -1
#
# im2 = Image.open('images/unmarked/train_cnn/Female Archback 4.jpg')
# img2 = np.array(im2.getdata()).reshape((size[1],size[0],3))
#
# (labels, values) = pool(img, img2, edge, 50)
# print(np.sum(labels)/len(labels))

file_arr = [filename for filename in os.listdir('images/unmarked/masks/')]
global_values = []
global_labels = []
for i in range(1):
    name = file_arr[i]
    im = Image.open('images/unmarked/masks/' + name)
    size = im.getdata().size
    img = np.array(im.getdata()).reshape((size[1], size[0], 3))
    img = img[:, :, 0]
    img[img > 0] = 1
    img[img <= 0] = -1

    im2 = Image.open('images/unmarked/train_cnn/' + name)
    img2 = np.array(im2.getdata()).reshape((size[1], size[0], 3))

    (labels, values) = pool(img, img2, edge, 50)
    global_values.append(values)
    global_labels.append(labels)

print(len(global_values))
