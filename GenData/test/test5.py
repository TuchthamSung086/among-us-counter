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


def write2D(arr, name):
    # Input: 2d numpy array, name of text file
    # Output: None
    # Write a file to show numpy data
    fout = open((name+".txt"), "w")
    s = ""
    fout.write("START\n")

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            s += " " + str(arr[i][j])
        s += "\n"
    # print(s)
    fout.write(s)

    fout.write("END\n")
    fout.close()


def pasteOn(fg, bg, size_multiply, rotate_angle, flip, mirror, x, y):
    # Input: Details of pasting
    # Output: New image of bg pasted with fg
    width, height = bg.size
    base_width = width/10

    fw, fh = fg.size
    fg = fg.resize((int(base_width*size_multiply), int(base_width/fw*fh*size_multiply)))
    fg = fg.rotate(rotate_angle, Image.NEAREST, expand=1)
    if flip:
        fg = ImageOps.flip(fg)
    if mirror:
        fg = ImageOps.mirror(fg)

    #print(width, height, x, y)
    loc_x = int(x*width)
    loc_y = int(y*height)
    bg.paste(fg, (loc_x, loc_y), fg.convert('RGBA'))
    return bg


def getFigure(PIL_image):
    # Input: Image imported from PIL
    # Output: Numpy where 1 is amongus and 0 is transparent
    rgb = red.convert("RGBA")
    arr = np.array(rgb)
    print("RGBA", arr.shape)
    area = np.zeros((rgb.size[1], rgb.size[0]))  # width, height for y, x im programming
    area[(arr != arr[0][0]).all(axis=2)] = 1
    return area


canteen = Image.open("bg/canteen.jpg")
canteen = canteen.convert("RGB")
red = Image.open("crewmate/red.png")
redArea = getFigure(red)
plt.imshow(redArea, cmap="gray")
plt.show()

print("red", np.array(red).shape)
print(np.array(red).min())
print(np.array(red).max())
#write2D(np.array(red), "red")
print("canteen", np.array(canteen).shape)

img = canteen
#img = pasteOn(red, canteen, 1, 0, 0, 0, 0, 0)

arr = np.array(img)  # (582, 666, 3)
print(arr.shape)
label = np.zeros(arr.shape[0:2], dtype="uint8")
# plt.imshow(label)
# plt.show()


def randomPaste():
    # Output: Random variables for pasting fg on bg
    sz = 1 + (random.random())*2
    rot = 360 * random.random()
    flip = random.randrange(0, 2)
    mirror = random.randrange(0, 2)
    x = random.random()
    y = random.random()
    return sz, rot, flip, mirror, x, y


for i in range(20):
    sz, rot, flip, mirror, x, y = randomPaste()
    #img = pasteOn(red, img, sz, rot, flip, mirror, x, y)
    img = pasteOn(red, img, sz, 0, 0, 0, x, y)
    label = pasteOn(red, img, sz, 0, 0, 0, x, y)

img.show()

plt.imshow(label, cmap='gray')
plt.show()
