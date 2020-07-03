import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
import statistics as stat
import sys
# Note that coord (relative to the image) needs to be centered at 0,0 for this to work
def get_circle(chrome_img):
    output = chrome_img.copy()
    cv.imshow("Img in get_circle", output)
    cv.waitKey(0)
    cv.destroyAllWindows()
    circles = cv.HoughCircles(output, cv.HOUGH_GRADIENT, 4, 400)
    if circles is None:
        print("Circles is none")
    if circles is not None:
        print("Size: ", len(circles))
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv.circle(output, (x, y), r, (100, 100, 100), 4)
            cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        cv.imshow("output in get_circle", np.hstack([chrome_img, output]))
        cv.waitKey(0)
    return circles

def find_chrome_reflect(chrome_img, circle):
    rad = int(circle[2]*0.9)
    inside_circle = []
    img = cv.GaussianBlur(chrome_img, (9, 9), 0) # Note that this value for Guassian blur 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist = np.sqrt((circle[0]-i)**2 + (circle[1]-j)**2)
            if dist < rad:
                #cv.circle(img, (i, j), 2, (100, 100, 0), 2)
                inside_circle.append([i, j, chrome_img[i][j]])
    # radius might need to be adjusted
    print("find chrome ref image size: ", img.shape)
    maxLocs = []
    inside_circle = np.transpose(inside_circle)
    print("inside_circle shape: ", np.shape(inside_circle[2]))

    for i in range(20):
        max_index = np.argmax(inside_circle[2])
        maxLoc = [inside_circle[0][max_index], inside_circle[1][max_index]]
        print("Max value! : ", inside_circle[:, max_index])
        print("Actual color: ", img[maxLoc[0]][maxLoc[1]])
        inside_circle = np.delete(inside_circle, max_index, 1)
        maxLocs.append(maxLoc)
        cv.circle(img, (maxLoc[0], maxLoc[1]), 10, (50, 0, 0), 4)
    cv.imshow('img in find_chrome_reflect', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    ret = np.mean(maxLocs, axis=0)
    return int(ret[0]), int(ret[1])

def find_sphere_normal(chrome_img, coord):
    # Frame of reference: Assume right hand coordinate system
    # coord[0] -= img.shape[0]/2
    # coord[1] -= img.shape[1]/2
    # We have information about the x and y axes, and we need to find the z-coord
    # for a complete normal vector
    zcoord = np.sqrt(radius**2 - (coord[0]-center[0])**2 - (coord[1]-center[1])**2)
    norm = [coord[0]-center[0], coord[1]-center[1], zcoord]
    magnitude = np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    norm = norm/magnitude
    return norm


dir_chrome = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/chrome_spheres"
chrome_img_files = [join(dir_chrome, f) for f in listdir(dir_chrome) if isfile(join(dir_chrome, f))]
N = []
R = [0, 0, 1]
L = []
print(chrome_img_files)
for file in chrome_img_files:
    chrome_img = cv.imread(file, 0) # Reads image in grayscale
    print("Chrome_img size: ", chrome_img.shape)
    circle = get_circle(chrome_img)[0]
    print("Circle: ", circle)
    center = [circle[0], circle[1]]
    radius = circle[2]
    cv.circle(chrome_img, (int(center[0]), int(center[1])), int(radius), (100, 0, 0), 2)
    cv.imshow('img with circle around', chrome_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    maxLoc = find_chrome_reflect(chrome_img, circle)
    print("MaxLoc: ", maxLoc)
    cv.circle(chrome_img, (maxLoc[0], maxLoc[1]), 50, (255, 0, 0), 2)

    N = (find_sphere_normal(chrome_img, maxLoc))
    #Maybe normalize?
    L.append([2*np.dot(N,R)*N[i]-R[i] for i in range(3)])
    

