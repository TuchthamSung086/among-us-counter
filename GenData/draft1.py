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
    # Output: 2D Numpy where 1 is amongus and 0 is transparent
    rgb = red.convert("RGBA")
    arr = np.array(rgb)
    print("RGBA", arr.shape)
    area = np.zeros((rgb.size[1], rgb.size[0]), dtype="uint8")  # width, height for y, x im programming
    area[(arr != arr[0][0]).all(axis=2)] = 255
    return area


def convertTransparentFromPIL(img, num):
    # Input: RGB Image
    # Output: Mark amongs with num, the rest is transparent
    img = img.convert("RGBA")

    datas = img.getdata()
    r, g, b = datas[0][0:3]
    newData = []

    for item in datas:
        if item[0] == r and item[1] == g and item[2] == b:  # transparent
            newData.append((num, num, num, 0))
        else:  # num
            newData.append((num, num, num, 255))

    img.putdata(newData)
    return img


def convertTransparentFromNumpy(arr):
    # Input: 2d numpy array of rgb image
    #       where 255 is for amongus
    #             0 is for transparent
    # Output: RGBA with transparent
    newData = []

    for row in arr:
        r = []
        for col in row:
            if col == 0:
                r.append((0, 0, 0, 0))
            else:
                r.append((col, col, col, 1))
        newData.append(r)
    img = Image.fromarray(np.array(newData))
    return img


def randomPaste():
    # Output: Random variables for pasting fg on bg
    sz = 1 + (random.random())*2
    rot = 360 * random.random()
    flip = random.randrange(0, 2)
    mirror = random.randrange(0, 2)
    x = random.random()
    y = random.random()
    return sz, rot, flip, mirror, x, y


# Import Images
canteen = Image.open("bg/canteen.jpg")
red = Image.open("crewmate/red.png")

# Generate


def gen1(fg, bg, it):
    # Initiate label
    label = Image.new("RGB", (bg.size))
    scale = 255
    minus = scale//it
    img = bg
    for i in range(it):
        sz, rot, flip, mirror, x, y = randomPaste()
        img = pasteOn(fg, img, sz, 0, 0, 0, x, y)
        figureImg = convertTransparentFromPIL(red, scale-minus*i)
        label = pasteOn(figureImg, label, sz, 0, 0, 0, x, y)
    return img, label


img1, label1 = gen1(red, canteen, 5)
img1.show()
label1.show()
