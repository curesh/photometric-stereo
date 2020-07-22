import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from os import getcwd

def main():
    dir_clean_img = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/my_data/obj8/IMG_20200721_105758.jpg"
    good_img = cv.imread(dir_clean_img, 0)
    # thresh = cv.threshold(good_img, 0, 255, cv.THRESH_OTSU)[1]
    thresh = cv.threshold(good_img, 105, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=8)
    
    kernel = np.ones((171,171),np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    mask = cv.bitwise_not(mask)
    cv.imshow("Gray", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    dir_img = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/my_data/obj8/"
    img_files = sorted([join(dir_img, f) for f in listdir(dir_img) if isfile(join(dir_img, f))])
    for file in img_files:
        img = cv.imread(file, 0)
        img_mask = cv.bitwise_or(img, img, mask=mask)
        cv.imshow("masked", img_mask)
        cv.imshow("orig", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
