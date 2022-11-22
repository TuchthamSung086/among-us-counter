# import
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import random
import cv2
from scipy import signal
import os
# from skimage.morphology import skeletonize, thin
import math
from PIL import Image, ImageOps
import random as random


img = cv2.imread("GenData/walking/idle.png")
# img = cv2.imread("GenData/crewmate/Yellow.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

print(np.unique(img, return_counts=True))


# edge 1 is body edge
# edge 2 is mask edge
# mask 1 2 3 is color, dark color and reflection
rgb_map = {'edge1': [1, 1, 35],
           'edge2': [2, 58, 2],
           'mask1': [0, 255, 0],
           'mask2': [0, 126, 0],
           'mask3': [252, 252, 252],
           'body': [255, 16, 16],
           'pants': [0, 0, 225],
           'shadow': [55, 59, 60]
           }

rgb_map3 = {'edge1': [1, 1, 35],
            'edge2': [2, 58, 2],
            'mask1': [0, 255, 0],
            'mask2': [0, 126, 0],
            'mask3': [252, 252, 252],
            'body': [255, 16, 16],
            'pants': [0, 0, 225],
            'shadow': [55, 59, 60]
            }

'''
imgg = Image.open("GenData/crewmate/Red.png")
imgg = np.array(imgg)
print(imgg[0][0])
print(np.unique(imgg, return_counts=True))
'''

idle_img = Image.open("GenData/crewmate/idle.png")
idle_img = idle_img.convert("RGB")
idle_arr = np.array(idle_img)
plt.imshow(idle_arr)
plt.show()
# print(arr[0][0])

# arr[arr== [255, 16, 16]][0:3] = [255, 255, 255]
# idx = np.array(arr == (rgb_map['shadow'])).all(axis=2)
# arr[idx] = [0, 0, 0]

# im = Image.fromarray(arr)
# im.save("GenData/crewmate/test.png")

img = cv2.imread("GenData/crewmate/test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()


def fillColor(img, r, g, b):
    # Input: RGB image, RGB that will replace r g b of original image
    # r = np.array(r)
    # g = np.array(g)
    # b = np.array(b)
    new = img.copy().astype("uint64")
    r0 = [255, 0, 0]  # 1*r + 0*g + 0*b
    g0 = [0, 255, 0]  # 0*r + 1*g + 0*b
    b0 = [0, 0, 255]  # 0*r + 0*g + 1*b

    rm = np.array(r0)/255
    gm = np.array(g0)/255
    bm = np.array(b0)/255
    matrix = np.array([[5153/7225,  10/17,	 41/85],
                       [13/867, 202/255,	 8/255],
                       [-97/65025,  13/15, 56/255]])
    # print(rm, gm, bm)
    for i in range(len(img)):
        for j in range(len(img[0])):
            # pixel = np.array([0, 0, 0])
            # r1, g1, b1 = img[i][j]
            # rr = r0*r
            # gg = g0*g
            # bb = b0*b

            new[i][j] = matrix @ img[i][j]
    print("NEW", type(new[0][0][0]))
    new[new > 255] = 255
    new[new < 0] = 0
    return new


new_idle = fillColor(idle_arr, 0, 0, 0)
plt.imshow(new_idle)
plt.show()
