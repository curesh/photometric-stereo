import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
import statistics as stat
import sys
from skimage import measure

# Note that coord (relative to the image) needs to be centered at 0,0 for this to work

def get_circle(chrome_img):
    output = chrome_img.copy()
    circles = cv.HoughCircles(output, cv.HOUGH_GRADIENT, 1.4, 400)
    if circles is None:
        print("Circles is none")
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv.circle(output, (x, y), r, (100, 100, 100), 4)
            cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
    return circles

def find_chrome_reflect(chrome_img, circle):
    rad = int(circle[2]*0.9)
    blurred = cv.GaussianBlur(chrome_img, (9, 9), 0) # Note that this value for Guassian blur 
    for i in range(blurred.shape[0]):
        for j in range(blurred.shape[1]):
            dist = np.sqrt((circle[0]-i)**2 + (circle[1]-j)**2)
            if dist > rad:
                blurred[i][j] = 0
    # radius might need to be adjusted
    maxLocs = []
    thresh = cv.threshold(blurred, 240, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=4)
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    
# loop over the unique components
    max_label = ()
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if (not max_label) or (max_label[0] <= numPixels):
            max_label = (numPixels, labelMask)
    if not max_label:
        print("Error no max label")
        return 0
    bright_patch = max_label[1]
    x_avg_center = 0
    y_avg_center = 0
    for i in range(bright_patch.shape[0]):
        for j in range(bright_patch.shape[1]):
            if bright_patch[i][j]:
                x_avg_center += j
                y_avg_center += i
    x_avg_center /= max_label[0]
    y_avg_center /= max_label[0]
    center_reflect = [int(x_avg_center), int(y_avg_center)]
    # cv.circle(chrome_img, (center_reflect[0], center_reflect[1]), 10, (130, 100, 100), 4)
    return center_reflect

def find_sphere_normal(chrome_img, coord, circle):
    # Frame of reference: Assume right hand coordinate system
    # We have information about the x and y axes, and we need to find the z-coord
    # for a complete normal vector
    zcoord = np.sqrt(circle[2]**2 - (coord[0]-circle[0])**2 - (coord[1]-circle[1])**2)
    norm = [coord[0]-circle[0], coord[1]-circle[1], zcoord]
    magnitude = np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    norm = norm/magnitude
    return norm

def main():
    dir_chrome = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/vani_data/chrome"
    chrome_img_files = [join(dir_chrome, f) for f in listdir(dir_chrome) if isfile(join(dir_chrome, f))]
    N = []
    R = [0, 0, 1]
    L = []
    for file in chrome_img_files:
        print(file)
        chrome_img = cv.imread(file, 0) # Reads image in grayscale
        if chrome_img.shape[0] > 500:
            scale = chrome_img.shape[0]/500
            width = int(chrome_img.shape[1] / scale)
            height = int(chrome_img.shape[0] / scale)
            dim = (width, height)
            chrome_img = cv.resize(chrome_img, dim, interpolation=cv.INTER_AREA)
        circle = get_circle(chrome_img)[0]
        print("Circle (x,y,r): ", circle)
        center = [circle[0], circle[1]]
        radius = circle[2]
        cv.circle(chrome_img, (int(center[0]), int(center[1])), int(radius), (100, 0, 0), 2)
        cv.circle(chrome_img, (int(center[0]), int(center[1])), 5, (100, 0, 0), 2)
        maxLoc = find_chrome_reflect(chrome_img, circle)
        if not maxLoc:
            continue
        print("Reflection point: ", maxLoc)
        cv.circle(chrome_img, (maxLoc[0], maxLoc[1]), 4, (140, 0, 0), 4)
        cv.imshow('chrome sphere post analysis', chrome_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        N = find_sphere_normal(chrome_img, maxLoc, circle)
        print("Normal vector: ")
        print(N)
        L_vector = [(2*np.dot(N,R)*N[i])-R[i] for i in range(3)]
        print("L vector: ")
        print(L_vector)
        #Maybe normalize?
        L.append(L_vector)
        

if __name__ == "__main__":
    main()


