import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
import statistics as stat

# Note that coord (relative to the image) needs to be centered at 0,0 for this to work
def get_circle(chrome_img):
    output = chrome_img.copy()
    circles = cv.HoughCircles(chrome_img, cv.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        cv2.imshow("output", np.hstack([chrome_img, output]))
        cv2.waitKey(0)

def find_chrome_ref(chrome_img):

    img = cv.GaussianBlur(chrome_img, (9, 9), 0) # Note that this value for Guassian blur 
    # radius might need to be adjusted
    # cv.imshow('img', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    (_, _, _, maxLoc) = cv.minMaxLoc(img)
    return maxLoc

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
    (center, radius) = get_circle(chrome_img)
    cv.circle(chrome_img, (int(center[0]), int(center[1])), int(radius), (100, 0, 0), 2)
    cv.imshow('img', chrome_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    maxLoc = find_chrome_ref(chrome_img)
    print("MaxLoc: ", maxLoc)
    cv.circle(chrome_img, maxLoc, 50, (255, 0, 0), 2)

    N = (find_sphere_normal(chrome_img, maxLoc))
    #Maybe normalize?
    L.append([2*np.dot(N,R)*N[i]-R[i] for i in range(3)])
    

