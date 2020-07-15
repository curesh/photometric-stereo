import cv2 as cv
import numpy as np
import argparse
import sys
from os import listdir
from os.path import isfile, join
import scipy
import os

def change_of_basis(S, z_coord, dim):
    z_norm = S[z_coord[0]*dim[0]+z_coord[1]]
    cb_matrix = [[1, 0, z_norm[0]],
                 [0, 1, z_norm[1]],
                 [0, 0, z_norm[2]]]
    cb_matrix_inv = np.linalg.inv(cb_matrix)
    S = np.array([cb_matrix_inv.dot(vect) for vect in S])
    return S

def main():
    parser = argparse.ArgumentParser(description="Perform unknown lighting photometric stereo on a dataset")
    parser.add_argument('-z', '--zloc',
                         help="Pixel coordinates in the image for the location where the surface normal is (0,0,1)")
    args = parser.parse_args(sys.argv[1:])
    print("start")
    dir_images = os.getcwd() + "test_data/cat/Objects"
    img_files = sorted([join(dir_images, f) for f in listdir(dir_images) if isfile(join(dir_images, f))])
    I = []
    for iterate in range(0, len(img_files), 3):
        file = img_files[iterate]
        print(file)
        img = cv.imread(file, 0)
        if img.shape[0] > 500:
            scale = img.shape[0]/500
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
            img = img[100:300, 50:300]
            dim = img.shape
        I.append(img.flatten())
    I = np.array(I).T
    print("I shape", I.shape)
    # Might have to tranpose first before svd
    u, sigma, vh = np.linalg.svd(I)
    print("42")
    u_clean = u[:,0:3]
    sigma_clean = sigma[0:3]
    vh_clean = vh[0:3]
    
    # NOTE that you also want to check the output if you negate all the sigma matrices to see if your righthandedness is correct
    s_hat = np.linalg.matmul(u_clean, np.sqrt(sigma_clean))
    l_hat = np.linalg.matmul(np.sqrt(sigma_clean), vh_clean)
    print("48")
    # 6 points to the right of 180, 90
    ind_albedo = [180*dim[0] + int(elem) for elem in np.linspace(90, 280, 6)]
    
    b_matrix_vars = np.zeros((len(ind_albedo), 6))
    ### Find 6 vectors in the rows of s_hat that are on the surface of the object.
    for i in range(len(ind_albedo)):
        s_i = s_hat[ind_albedo[i]]
        b_matrix_vars[i] = [s_i[0]**2, 2*s_i[0]*s_i[1], 2*s_i[0]*s_i[2], s_i[1]**2, 2*s_i[1]*s_i[2], s_i[2]**2, 1]
    b = np.linalg.solve(b_matrix_vars[:,:-1], b_matrix_vars[:, -1:])
    b = np.array([[b[0], b[1], b[2]],
                  [b[1], b[3], b[4]],
                  [b[2], b[4], b[5]]])
    b_u, b_sigma = np.linalg.svd(b)
    A = np.linalg.matmul(b_u, np.sqrt(b_sigma))
    print("63")
    S = np.linalg.matmul(s_hat, A)
    L = np.linalg.matmul(np.linalg.inv(A), l_hat)
    
    # You need to find a pixel where the real surface norm is (0,0,1): This is 300, 225 in the harvard dataset
    z_coord = (int(args.z.split(",")[0]), int(args.z.split(",")[1]))
    S = change_of_basis(S, z_coord, dim)
    S = np.reshape(S, (dim[0], dim[1]))
    print("71")
    cv.imshow("Colormap", S)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()