import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
import statistics as stat
import sys
from skimage import measure

# This function finds the chrome sphere location in a chrome sphere img
def get_circle(chrome_img):
    circles = cv.HoughCircles(chrome_img, cv.HOUGH_GRADIENT, 2, 400)
    if circles is None:
        print("Circles is none")
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # show the output image
    return circles

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
    thresh = cv.threshold(blurred, 240, 255, cv.THRESH_BINARY)[1]
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

# This function prints out the RMS errors and other statistics of the surface normal computation
def get_errors(albedo, surface_normals, I, masks, L):
    Ierr = np.array([])
    # This assumes that each normal vector is a row, and that image matrix is flattened into an array
    b = np.array([vect*albedo[i] for i,vect in enumerate(surface_normals)])
    print("albedo shape ", albedo.shape, " surface_normals shape: ", surface_normals.shape, " I shape:", I.shape, " masks shape: ", masks.shape, " L shape: ", L.shape, " b shape: ", b.shape)
    for i in range(I.shape[1]):
        Ierri = np.array(I[:,i] - [np.dot(b[j], L[i]) for j in range(b.shape[0])])
        for iterate, element in enumerate(Ierri):
            if not masks[i,iterate]:
                Ierri[iterate] = 0
        if not Ierr.size:
            Ierr = Ierri**2
        else:
            Ierr += Ierri**2
    isfin = np.isfinite(np.sqrt(Ierr/ np.sum(masks, 0)))

    Ierr = [element for i, element in enumerate(Ierr) if isfin[i]]
    Ierr = np.array(Ierr)
    print("Evaluate scaled normal estimation by intensity error:")
    print("RMS: ", np.sqrt(np.mean(Ierr**2)))
    print("Mean: ", np.mean(Ierr))
    print("Median: ", np.median(Ierr))
    print("90 percentile: ", np.percentile(Ierr, 90))
    print("Max: ", np.max(Ierr))

#This function does the chrome_sphere analysis and returns the light direction matrix
def chrome_sphere_analysis(dir_chrome):
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
        center = [circle[0], circle[1]]
        radius = circle[2]
        cv.circle(chrome_img, (int(center[0]), int(center[1])), int(radius), (100, 0, 0), 2)
        cv.circle(chrome_img, (int(center[0]), int(center[1])), 5, (100, 0, 0), 2)
        max_loc = find_chrome_reflect(chrome_img, circle)
        if not max_loc:
            continue
        cv.circle(chrome_img, (max_loc[0], max_loc[1]), 4, (140, 0, 0), 2)
        chrome_img = draw_gridlines(chrome_img)
        
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
    for count, file in enumerate(img_files):
        # if count == 4:
        #     break
        print(file)
        img = cv.imread(file, 0)
        if img.shape[0] > 500:
            scale = img.shape[0]/500
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        curr_I = [element for row in img for element in row]
        I.append(curr_I)
        maski = cv.threshold(img, 26, 255, cv.THRESH_BINARY)[1]
        kernel = np.ones((5,5),np.uint8)
        maski = cv.erode(maski,kernel,iterations = 1)
        maski = maski.flatten()
        print("maski size!!: ", maski.shape, " img size: ", img.shape)
        masks.append(maski)
    masks = np.array(masks)
    I = np.array(I)
    G = get_surface_normals(L, I).T
    print("G shape: ", G.shape)
    albedo = np.array([np.linalg.norm(vect) for vect in G])
    surface_normals_flat = np.array([vect/np.linalg.norm(vect) for vect in G]).T
    
    get_errors(albedo, surface_normals_flat.T, I.T, masks, L)
    
    surface_normals_flat = np.array([vect/np.linalg.norm(vect) for vect in G]).T
    print("Output shape: ", surface_normals_flat.shape)
    surface_normals = []
    for pixels in surface_normals_flat:
        print("Pixels size: ", len(pixels))
        arr = np.reshape(pixels, (dim[1], dim[0]))
        surface_normals.append(arr)
    surface_normals = np.array(surface_normals)
    print("Final surface normals shape: ", surface_normals.shape)
    row = []
    r = np.array(surface_normals[0])
    g = np.array(surface_normals[1])
    b = np.array(surface_normals[2])
    surface_normals = cv.merge((b, g, r))
    return surface_normals

def main():
    dir_chrome = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/cat/LightProbe-1"
    L = chrome_sphere_analysis(dir_chrome)
    dir_img = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/cat/Objects"
    surface_normals = pms_analysis(dir_img, L)
    print("final shape: ", surface_normals.shape)
    cv.imshow("Colormap", surface_normals)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()