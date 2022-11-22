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
    bg = bg.copy()
    fg = fg.copy()
    width, height = bg.size
    base_width = width/10

    fw, fh = fg.size
    if flip:
        fg = ImageOps.flip(fg)
    if mirror:
        fg = ImageOps.mirror(fg)
    #print("After flip,mirror", np.unique(fg, return_counts=True))
    fg = fg.resize((int(base_width*size_multiply), int(base_width/fw*fh*size_multiply)), Image.Resampling.NEAREST)
    #print("After resize", np.unique(fg, return_counts=True))
    fg = fg.rotate(rotate_angle, Image.Resampling.NEAREST, expand=1)
    #print("After rotate", np.unique(fg, return_counts=True))

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
    #print("RGBA", arr.shape)
    area = np.zeros((rgb.size[1], rgb.size[0]), dtype="uint8")  # width, height for y, x im programming
    area[(arr != arr[0][0]).all(axis=2)] = 255  # that pixel looks like background
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

    #print("NEWDATA UNIQUE", np.unique(newData, return_counts=True))
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
    sz = 1 + (random.random())
    rot = 360 * random.random()
    flip = random.randrange(0, 2)
    mirror = random.randrange(0, 2)
    x = random.random()
    y = random.random()
    return sz, rot, flip, mirror, x, y


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


def makeDictFromLists(test_keys, test_values):
    test_keys = list(test_keys)
    test_values = list(test_values)
    res = {}
    for key in test_keys:
        for value in test_values:
            res[key] = value
            test_values.remove(value)
            break
    return res


def isImgOverlap(img1, img2):
    # img1 is before
    # img2 is after
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    cnt1 = np.unique(arr1, return_counts=True)
    cnt2 = np.unique(arr2, return_counts=True)
    dic1 = makeDictFromLists(cnt1[0], cnt1[1])
    dic2 = makeDictFromLists(cnt2[0], cnt2[1])
    for key in dic1:
        if key != 0 and key in dic2 and dic1[key] > dic2[key]:
            return True
    return False


def isInBoundary(fg, bg, sz, rot, flip, mirror, x, y):
    width, height = bg.size
    fwidth, fheight = fg.size
    cx = int(width / 2)
    cy = int(height / 2)
    thetha = rot

    rad = math.radians(thetha)
    base_width = width/10
    fg = fg.resize((int(base_width*sz), int(base_width/fwidth*fheight*sz)))
    fwidth, fheight = fg.size
    loc_x = int(x*width)
    loc_y = int(y*height)
    corners = [[loc_x, loc_y], [loc_x+fwidth, loc_y], [loc_x, loc_y+fheight], [loc_x+fwidth, loc_y+fheight]]
    newCorners = []
    for px, py in corners:
        new_px = cx + int(float(px-cx) * math.cos(rad) + float(py-cy) * math.sin(rad))
        new_py = cy + int(-(float(px-cx) * math.sin(rad)) + float(py-cy) * math.cos(rad))
        #print(px, py, new_px, new_py, width, height)
        if new_px > width or new_px < 0:
            return False
        if new_py > height or new_py < 0:
            return False
    return True


def gen2(fg, bg, it):
    # Same Size, No rotate, No flip, No mirror, No overlap, No out of edge
    # Initiate label
    label = Image.new("RGB", (bg.size))
    scale = 255
    minus = scale//it
    img = bg
    cnt = 0
    for i in range(it):
        sz, rot, flip, mirror, x, y = randomPaste()
        figureImg = convertTransparentFromPIL(red, scale-minus*i)
        if not isInBoundary(figureImg, label, sz, rot, flip, mirror, x, y):
            #print("OUT OF BOUNDARY")
            continue
        labelnew = pasteOn(figureImg, label, sz, rot, flip, mirror, x, y)
        if isImgOverlap(label, labelnew):
            # print("OVERLAP")
            continue
        img = pasteOn(fg, img, sz, rot, flip, mirror, x, y)
        label = labelnew
        cnt += 1
    return img, label, cnt


# Import Images
canteen = Image.open("GenData/bg/canteen.jpg")
red = Image.open("GenData/crewmate/red.png")

'''
img1, label1, cnt = gen2(red, canteen, 10)
img1.show()
label1.show()
print(cnt)
'''
###### DONT TOUCH ANYTHING ABOVE THIS #####

# CHANGE PARAMETERS BELOW AND RUN
fout = open("GenData/set5/count.txt", "w", buffering=1)
for i in range(10000):
    tries = random.randint(3, 20)
    img, lab, cnt = gen2(red, canteen, tries)
    img.save("GenData/set5/img"+str(i)+".jpg")
    lab.save("GenData/set5/label"+str(i)+".png")
    fout.write(str(i)+" "+str(cnt)+"\n")
    fout.flush()
    print(str(i)+" images done")
fout.close()
