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
#from skimage.morphology import skeletonize, thin
import math
from PIL import Image, ImageOps


def main():
    return

# Input parameters
# fg: foreground image imported from Image.open("bg/canteen.jpg")
# bg: background image imported from Image.open("...
# size_multiply: size of fg
# x,y: between -0.5 to 0.5 to place at random positions


def pasteOn(fg, bg, size_multiply, rotate_angle, flip, mirror, x, y):
    width, height = bg.size
    base_width = width/10

    fw, fh = fg.size
    fg = fg.resize((int(base_width*size_multiply), int(base_width/fw*fh*size_multiply)))
    fg = fg.rotate(rotate_angle, Image.NEAREST, expand=1)
    if flip:
        fg = ImageOps.flip(fg)
    if mirror:
        fg = ImageOps.mirror(fg)

    loc_x = int(width/2 + x*width)
    loc_y = int(height/2 + y*height)
    bg.paste(fg, (loc_x, loc_y), fg.convert('RGBA'))
    return bg
