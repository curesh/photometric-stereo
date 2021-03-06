from os import listdir
from os.path import isfile, join
from os import getcwd
import sys
import cv2 as cv
import numpy as np
from skimage import measure
import argparse
import math
import matplotlib.pyplot as plt

# This function finds the chrome sphere location in a chrome sphere img
def get_mask():
    dir_clean_img = "./test_data/my_data/obj8/IMG_20200721_105758.jpg"
    good_img = cv.imread(dir_clean_img, 0)
    # thresh = cv.threshold(good_img, 0, 255, cv.THRESH_OTSU)[1]
    thresh = cv.threshold(good_img, 105, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=8)
    
    kernel = np.ones((171,171),np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    mask = cv.bitwise_not(mask)
    return mask

def get_circle(chrome_img, dp):
    circles = cv.HoughCircles(chrome_img, cv.HOUGH_GRADIENT, dp, 400)
    if circles is None:
        print("Circles is none")
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles[0]

# This function finds the location of the glare in the chrome sphere img
def find_chrome_reflect(chrome_img, circle):
    rad = int(circle[2]*0.9)
    blurred = cv.GaussianBlur(chrome_img, (9, 9), 0) # Note that this value for Guassian blur 
    # print("Blurred shape: ", blurred.shape)
    for i in range(blurred.shape[0]):
        for j in range(blurred.shape[1]):
            dist = np.sqrt((circle[1]-i)**2 + (circle[0]-j)**2)
            if dist > rad:
                blurred[i][j] = 0
    # radius might need to be adjusted
    thresh = cv.threshold(blurred, 230, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=4)
    labels = measure.label(thresh, connectivity=2, background=0)
    
# loop over the unique components
    max_label = ()
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        label_mask = np.zeros(thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv.countNonZero(label_mask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if (not max_label) or (max_label[0] <= num_pixels):
            max_label = (num_pixels, label_mask)
    if not max_label:
        sys.exit("Error no max label")
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
    return center_reflect

# This function returns the normal vector of the surface of a sphere, given the x,y coordinates
def find_sphere_normal(coord, circle):
    # Frame of reference: Assume right hand coordinate system
    # We have information about the x and y axes, and we need to find the z-coord
    # for a complete normal vector
    zcoord = np.sqrt(circle[2]**2 - (coord[0]-circle[0])**2 - (coord[1]-circle[1])**2)
    norm = [coord[0]-circle[0], -1 * (coord[1]-circle[1]), zcoord]
    magnitude = np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    norm = norm/magnitude
    return norm

# This function draws gridlines on an image
def draw_gridlines(chrome_img):
    width = chrome_img.shape[0]
    height = chrome_img.shape[1]
    for i in range(0, height, 50):
        cv.line(chrome_img, (i, 0), (i, width), (0, 0, 0), 1)
    for i in range(0, width, 50):
        cv.line(chrome_img, (0, i), (height, i), (0, 0, 0), 1)
    return chrome_img

# This function does the lambarts law computations to get the G matrix (surface normals and albedo)
def get_surface_normals(L, I):
    G = np.matmul(L.T, I)
    inv = np.linalg.inv(np.matmul(L.T, L))
    G = np.matmul(inv, G)
    return G

def compare_harvard_sn(surface_normals, final_mask):
    my_sn = surface_normals.copy()
    cv.imshow("just orig", surface_normals)
    harvard_sn_dir = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/cat/original_sn.png"
    orig_my_img_dir = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/cat/orig_my_img.png"
    my_sn = cv.imread(orig_my_img_dir, 1)
    harvard_sn = cv.imread(harvard_sn_dir, 1)
    avg_mae = 0
    scale = harvard_sn.shape[0]/500
    width = int(harvard_sn.shape[1] / scale)
    height = int(harvard_sn.shape[0] / scale)
    dim = (width, height)
    harvard_sn = cv.resize(harvard_sn, dim, interpolation=cv.INTER_AREA)
    error_matrix = np.zeros(harvard_sn.shape)
    for i in range(harvard_sn.shape[0]):
        for j in range(harvard_sn.shape[1]):
            if final_mask[i][j]:
                mags = round(np.linalg.norm(my_sn[i][j])*np.linalg.norm(harvard_sn[i][j]))
                norm = np.dot(np.double(my_sn[i][j]), np.double(harvard_sn[i][j]))
                norm = norm/mags
                avg_mae += abs(math.acos(norm))
                error_matrix[i][j] = abs(math.acos(norm))
            else:
                my_sn[i][j] = [255,255,255]
    error_matrix = np.uint8(error_matrix*256)
    # my_sn = cv.normalize(my_sn, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    # my_sn = np.uint8(my_sn) 
    cv.imshow("orig with change", my_sn)
    cv.imshow("harvard", harvard_sn)
    cv.imshow("error matrix", error_matrix)
    cv.imshow("diff", np.absolute(harvard_sn-my_sn[:, :]))
    cv.waitKey(0)
    avg_mae /= np.count_nonzero(final_mask)
    avg_mae *= 360/(2*math.pi)
    print("Average mean angle error: ", avg_mae)
    print("median of mine", np.median(my_sn))
    print("median of harvard", np.median(harvard_sn))

# This function prints out the RMS errors and other statistics of the surface normal computation
def get_errors(albedo, surface_normals, I, masks, L, dim):
    Ierr = np.array([])
    # This assumes that each normal vector is a row, and that image matrix is flattened into an array
    b = np.array([vect*albedo[i] for i,vect in enumerate(surface_normals)])

    for i in range(I.shape[1]):
        reconstructed = np.reshape(np.array([np.dot(b[j], L[i]) if np.dot(b[j], L[i])>0 else 0 for j in range(b.shape[0])]), (dim[1], dim[0]))
        # THe following code is to look at the reconstructed images
        # reconstructed = np.uint8(reconstructed)
        # print("IMportant orig image dtype", np.reshape(I[:, i], (dim[1], dim[0])).dtype)
        # print("IMportant reconstucted image dtype", reconstructed.dtype)

        # cv.imshow("orig", np.reshape(I[:,i], (dim[1], dim[0])))
        # cv.imshow("reconstructed", reconstructed)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # reconstructed = [np.single(elem) for elem in reconstructed.flatten()]
        reconstructed = reconstructed.flatten()
        
        
        Ierri = np.array(np.double(I[:,i]) - reconstructed)/255

        #NOTE Uncomment once masks starts working
        
        # for iterate, element in enumerate(Ierri):
        #     if not masks[i,iterate]:
        #         Ierri[iterate] = 0
        if not Ierr.size:
            Ierr = Ierri**2
        else:
            Ierr += Ierri**2
    # with np.errstate(divide='ignore',invalid='ignore'):
    #     isfin = np.isfinite(np.sqrt(Ierr/ np.sum(masks, 0)))
    isfin = np.isfinite(np.sqrt(Ierr/ I.shape[1]) )
    Ierr = [element for i, element in enumerate(Ierr) if isfin[i]]
    Ierr = np.array(Ierr)
    print("Evaluate scaled normal estimation by intensity error:")
    print("RMS: ", np.sqrt(np.mean(Ierr**2)))
    print("Mean: ", np.mean(Ierr))
    print("Median: ", np.median(Ierr))
    print("90 percentile: ", np.percentile(Ierr, 90))
    print("Max: ", np.max(Ierr))
    plt.hist(Ierr.flatten(), np.linspace(0, 0.012, 10))
    plt.xlabel("Difference between ground truth and calculated intensity for each pixel")
    plt.title("Root mean square error per pixel histogram")
    plt.show()

#This function does the chrome_sphere analysis and returns the light direction matrix
def chrome_sphere_analysis(dir_chrome, args):
    chrome_img_files = sorted([join(dir_chrome, f) for f in listdir(dir_chrome) if isfile(join(dir_chrome, f))])
    N = []
    R = [0, 0, 1]
    L = []
    circle = [625, 597, 528]
    for count, file in enumerate(chrome_img_files):
        # if count == 4:
        #     break
        print(file)
        chrome_img = cv.imread(file, 0) # Reads image in grayscale
        scale = 1
        if chrome_img.shape[0] > 500:
            scale = chrome_img.shape[0]/500
            width = int(chrome_img.shape[1] / scale)
            height = int(chrome_img.shape[0] / scale)
            dim = (width, height)
            chrome_img = cv.resize(chrome_img, dim, interpolation=cv.INTER_AREA)
            # cv.imshow("Original chrome sphere", chrome_img)
        if args.toggle:
            circle = [625, 597, 528]
            circle = [element/scale for element in circle]
        else:
            circle = get_circle(chrome_img, args.dp)

        center = [circle[0], circle[1]]
        radius = circle[2]
        cv.circle(chrome_img, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 2)
        cv.circle(chrome_img, (int(center[0]), int(center[1])), 5, (255, 0, 0), 2)
        # cv.imshow("Chrome sphere outlined", chrome_img)
        max_loc = find_chrome_reflect(chrome_img, circle)
        if not max_loc:
            continue
        cv.circle(chrome_img, (max_loc[0], max_loc[1]), 4, (0, 0, 0), 2)
        
        # Uncomment if you want gridlines
        # chrome_img = draw_gridlines(chrome_img)

        # Main display for chrome sphere
        # cv.imshow('chrome sphere post analysis', chrome_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        N = find_sphere_normal(max_loc, circle)
        L_vector = [(2*np.dot(N,R)*N[i])-R[i] for i in range(3)]
        #Maybe normalize?
        L.append(L_vector)
    L = np.array(L)
    L = np.array([vect/np.linalg.norm(vect) for vect in L])
    return L

# This function does the photometric stereo analysis and returns the surface normal matrix
def pms_analysis(dir_img, L):
    img_files = sorted([join(dir_img, f) for f in listdir(dir_img) if isfile(join(dir_img, f))])
    I = []
    masks = []
    #mask = get_mask()
    for count, file in enumerate(img_files):
        # if count == 4:
        #     break
        img = cv.imread(file, 0)
        #img = cv.bitwise_or(img, img, mask=mask)
        if img.shape[0] > 500:
            scale = img.shape[0]/500
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        curr_I = [element for row in img for element in row]
        I.append(curr_I)
        maski = cv.threshold(img, 26, 255, cv.THRESH_BINARY)[1]
        kernel = np.ones((20,20),np.uint8)
        maski = cv.erode(maski,kernel,iterations = 1)
        maski = maski.flatten()
        masks.append(maski)
    masks = np.array(masks)
    final_mask = np.uint8(np.reshape(np.sum(masks, axis=0), (dim[1], dim[0])))
    print("final mask shape", final_mask.shape)
    final_mask = cv.threshold(final_mask, 1, 255, cv.THRESH_BINARY)[1]
    # cv.imshow("Mask", final_mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    np.savetxt("final_mask.txt", final_mask, delimiter=',')
    I = np.array(I)
    G = get_surface_normals(L, I).T
    print("G shape: ", G.shape)
    albedo = np.array([np.linalg.norm(vect) for vect in G])
    surface_normals_flat = np.array([vect/np.linalg.norm(vect) for vect in G]).T
    print("I.T shape", I.T.shape)
    get_errors(albedo, surface_normals_flat.T, I.T, masks, L, dim)
    
    surface_normals = []
    for pixels in surface_normals_flat:
        arr = np.reshape(pixels, (dim[1], dim[0]))
        surface_normals.append(arr)
    surface_normals = np.array(surface_normals)
    print("Final surface normals shape: ", surface_normals.shape)
    r = np.array(surface_normals[0])
    g = np.array(surface_normals[1])
    b = np.array(surface_normals[2])
    surface_normals = cv.merge((b, g, r))
    
    #compare_harvard_sn(surface_normals, final_mask)
    return surface_normals

def main():
    parser = argparse.ArgumentParser(description="Perform photometric stereo on a dataset")
    parser.add_argument('--dp', default=2, type=float,
                         help="dp parameter for HoughCircles function: pick a value between 1 and 5 (2 is ideal)")
    parser.add_argument('-t', '--toggle', action="store_true",
                        help="Toggle this option if you are operating on the harvard dataset")
    args = parser.parse_args(sys.argv[1:])
    if args.toggle:
        dir_chrome = getcwd() + "/test_data/cat/LightProbe-1"
        dir_img = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/cat/Objects"
    else:
        dir_chrome = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/my_data/chrome2"
        dir_img = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/my_data/pisa"

    L = chrome_sphere_analysis(dir_chrome, args)
    surface_normals = pms_analysis(dir_img, L)
    print("final shape: ", surface_normals.shape)
    cv.imshow("Colormap", surface_normals)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()
