#square
#shouldn't be larger than 800x800


import cv2
import numpy as np
import os

def find_percent_transparent(image):
    return 100 - np.average(image[:,:,3])*100/255

def random_square_background(dir_name,size,N = 1,threshold_percent = 10):
    l = []
    bglist = os.listdir(dir_name)
    count_fail = 0
    while len(l) < N:
        if count_fail > max(20,2*N):
            raise Exception('Fail to generate background corresponding to param')
        image_name = np.random.choice(bglist)
        print(image_name)
        background = cv2.imread(f'{dir_name}\\{image_name}',cv2.IMREAD_UNCHANGED)
        if size == -1:
            size = np.random.randint(100,800)
        elif type(size) == tuple:
            if len(size) != 2:
                raise Exception('You can only use a tuple with two integers!')
            else:
                size = np.random.randint(size[0],size[1])
        

        if background.ndim == 2:
            height, width = background.shape
        else:
            height, width, nchannel = background.shape
        if height > size or width > size:
            count_fail += 1
            continue
        x = np.random.randint(0,width-size+1)
        y = np.random.randint(0,height-size+1)

        patch = background[y:y+size,x:x+size]

        if find_percent_transparent(patch) < threshold_percent:
            l.append(patch)
            count_fail = 0
        else:
            count_fail += 1

    if N == 1:
        return l[0]
    return l

###### TEST CODE #####
# import matplotlib.pyplot as plt
# background = cv2.imread('Background\\The_Skeld_Admin.png',cv2.IMREAD_UNCHANGED)
# l = random_square_background('Background',200,6)

# fig,ax = plt.subplots(6)
# for i in range(6):    
#     ax[i].imshow(l[i])
# plt.show()
