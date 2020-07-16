import cv2 as cv
import numpy as np
import argparse
import sys
from os import listdir
from os.path import isfile, join
import scipy
import os
from PIL import Image

def change_of_basis(S, z_coord, dim):
    z_norm = S[z_coord[0]*dim[1]+z_coord[1]]
    y_norm = S[53*dim[1]+102]
    cb_matrix = [[1, 0, z_norm[0]],
                 [0, 1, z_norm[1]],
                 [0, 0, z_norm[2]]]
    cb_matrix_inv = np.linalg.inv(np.array(cb_matrix))
    S = np.array([cb_matrix_inv.dot(vect) for vect in S])
    return S

def main():
    # PARAMS
    height_scale = 0.5 #scale relative to 500
    
    parser = argparse.ArgumentParser(description="Perform unknown lighting photometric stereo on a dataset")
    parser.add_argument('-z', '--zloc', required=True,
                         help="Pixel coordinates in the image for the location where the surface normal is (0,0,1)")
    args = parser.parse_args(sys.argv[1:])
    print("start")
    dir_images = os.getcwd() + "/test_data/cat/Objects"
    img_files = sorted([join(dir_images, f) for f in listdir(dir_images) if isfile(join(dir_images, f))])
    I = []
    # You need to find a pixel where the real surface norm is (0,0,1): This is 300, 225 in the harvard dataset: 200, 175 in adjusted
    z_coord = (int(args.zloc.split(",")[0]), int(args.zloc.split(",")[1]))
    print("Z-cord: ", z_coord)
    for iterate in range(0, len(img_files), 3):
        file = img_files[iterate]
        print(file)
        img = cv.imread(file, 0)
        if img.shape[0] > 500*height_scale:
            scale = img.shape[0]/(500*height_scale)
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
            img = img
            dim = img.shape
        I.append(img.flatten())
    I = np.array(I).T
    dim = (dim[0], dim[1], I.shape[1])
    print("dim shape", dim)
    # Might have to tranpose first before svd
    u, sigma, vh = np.linalg.svd(I)
    print("u shape: ", u.shape)
    print("sigma shape: ", sigma.shape)
    print("vh shape: ", vh.shape)
    print("42")
    u_clean = u[:,0:3]
    sigma = sigma[0:3]
    vh_clean = vh[0:3]
    sigma_clean = np.diag(sigma)
    print("u_clean shape", u_clean.shape)
    print("sigma_clean shape", sigma_clean.shape)
    print("vh_clean shape", vh_clean.shape)
    # NOTE that you also want to check the output if you negate all the sigma matrices to see if your righthandedness is correct
    s_hat = np.matmul(u_clean, np.sqrt(sigma_clean))
    l_hat = np.matmul(np.sqrt(sigma_clean), vh_clean)
    print("48")
    print("s_hat shape ", s_hat.shape)
    print("l_hat shape ", l_hat.shape)
    # 6 points to the right of 180, 90
    ind_albedo = [int(180*height_scale)*dim[1] + int(elem) for elem in np.linspace(90*height_scale, 280*height_scale, 6)]
    print("ind albedo: ", ind_albedo)
    b_matrix_vars = np.zeros((len(ind_albedo), 7))
    ### Find 6 vectors in the rows of s_hat that are on the surface of the object.
    for i in range(len(ind_albedo)):
        s_i = s_hat[ind_albedo[i]]
        b_matrix_vars[i] = [s_i[0]**2, 2*s_i[0]*s_i[1], 2*s_i[0]*s_i[2], s_i[1]**2, 2*s_i[1]*s_i[2], s_i[2]**2, 1]

    b = np.linalg.lstsq(b_matrix_vars[:,:-1], b_matrix_vars[:, -1:])[0]
    print("original b shape: ", b.shape)
    print("original b:", b.shape)
    b = np.array([[b[0], b[1], b[2]],
                  [b[1], b[3], b[4]],
                  [b[2], b[4], b[5]]])[:,:,0]
    print("b shape: ", b.shape)
    print("b: ", b)
    b_u, b_sigma, b_vh = np.linalg.svd(b)
    print("b_u shape: ", b_u.shape)
    print("b_sigma: ", b_sigma.shape)
    b_sigma = np.diag(b_sigma)
    print(b_sigma)
    A = np.matmul(b_u, np.sqrt(b_sigma))
    print("63")
    print("A shape: ", A.shape)
    S = np.matmul(s_hat, A)
    L = np.matmul(np.linalg.inv(A), l_hat)
    print("S shape: ", S.shape)
    print("L shape: ", L.shape)

    S = change_of_basis(S, z_coord, dim)
    print("Check S before adjust to int8: ", S[dim[1]*z_coord[0] + z_coord[1]])
    print("Confirmation value after normalization: ", np.uint8( 128* (S[dim[1]*z_coord[0]+z_coord[1]]+1)))
    #    S = S.clip(min=0)
    #S = np.array([np.uint8(128*(vect/np.linalg.norm(vect)+1)) for vect in S])
    S = np.array([vect/np.linalg.norm(vect) for vect in S])
    print("Other check for below purpose: ", S[dim[1]*z_coord[0]+z_coord[1]])
    print("Other check for below purpose: ", S[0])
    surface_normals = np.reshape(S, (dim[0], dim[1], 3))
    print("surface_normals shape: ", surface_normals.shape)

    print("This vector is the transformed z vector check. It should be 0,0,1: ", surface_normals[z_coord[0]][z_coord[1]])
    print("This vector is the transformed z vector check. It should be 0,0,1: ", surface_normals[0][0])
    print("71")

    np.savetxt('output.txt', S, delimiter=',')
    cv.imwrite("output.jpg", surface_normals)
if __name__ == "__main__":
    main()
