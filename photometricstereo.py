import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

dir_chrome = ""
chrome_imgs = [f for f in listdir(dir_chrome) if isfile(join(dir_chrome, f))]

#Note that coord (relative to the image) needs to be centered at 0,0 for this to work

def find_sphere_normal(chrome_img, radius, coord):
    # Frame of reference: Assume that the center of the sphere is the origin and 
    # right hand coordinate system
    edges = cv2.Canny(chrome_img, img,100,200)
    center = np.mean(edges, axis=1) # axis might be 0
    # We have information about the x and y axes, and we need to find the z-coord
    # for a complete normal vector
    norm = (coord[0], coord[1], np.sqrt(coord[0]**2 + coord[1]**2))
    return norm