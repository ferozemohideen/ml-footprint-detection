# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import cv2
#
# fig,axes = plt.subplots(1,2)
# template = cv2.imread('images/unmarked/masks/Female Archback 4.jpg')
# rows, cols, _ = template.shape
#
# pts1 = np.float32([[177,162],[337,150],[441,302]])
# pts2 = np.float32([[307,127],[520,165],[558,312]])
#
# M = cv2.getAffineTransform(pts1,pts2)
#
# dst = cv2.warpAffine(template,M,(cols,rows), borderValue=(255,255,255))
# axes[0].imshow(template)
# axes[1].imshow(dst)
# plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt
from test_image_blur import detectEdges

img1 = cv2.imread('images/unmarked/train_cnn/Female Archback 4.jpg', 0)          # queryImage
#img1 = detectEdges(img1)
img2 = cv2.imread('images/unmarked/train_cnn/Female Archback 4.jpg',0) # trainImage
#img2 = detectEdges(img2)

# Initiate SIFT detector
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

plt.imshow(img3),plt.show()