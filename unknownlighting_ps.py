import cv2 as cv
import numpy as np
import argparse
import sys
from os import listdir
from os.path import isfile, join
import scipy
import os
import math

def get_mask():
    dir_clean_img = "./test_data/my_data/pisa/IMG_20200721_232857.jpg"
    good_img = cv.imread(dir_clean_img, 0)

    thresh = cv.threshold(good_img, 0, 255, cv.THRESH_OTSU)[1]
    # thresh = cv.threshold(good_img, 90, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)
    thresh = cv.bitwise_not(thresh)
    contour,hier = cv.findContours(thresh,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(c) for c in contour]
    max_index = np.argmax(areas)
    cnt=contour[max_index]
    cv.drawContours(thresh,[cnt],0,255,-1)
    gray = cv.bitwise_not(thresh)
    kernel = np.ones((9,9),np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

    return mask

def compare_harvard_sn(final_mask):
    harvard_sn_dir = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/test_data/cat/original_sn.png"
    orig_my_img_dir = "/Users/bigboi01/Documents/CSProjects/KadambiLab/photometricStereo/my_img_ul.png"
    my_sn = cv.imread(orig_my_img_dir, 1)
    harvard_sn = cv.imread(harvard_sn_dir, 1)
    print("harvard shape: ", harvard_sn.shape)
    print("my_sn: ", my_sn.shape)
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
                if norm == 0 and mags == 0:
                    norm = 0
                else:
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

def change_of_basis(S, z_coord, y_coord, x_coord, dim):
    print("Y_coord in change of basis: ", y_coord)
    print("Dim in change of bas: ", dim)
    print("Calculation: ", int(y_coord[0]*dim[1]+y_coord[1]))
    z_norm = S[int(z_coord[0]*dim[1]+z_coord[1])]
    y_norm = S[int(y_coord[0]*dim[1]+y_coord[1])]
    x_norm = S[int(x_coord[0]*dim[1]+x_coord[1])]
    print("Y-norm!!, ", y_norm)
    cb_matrix = [[x_norm[0], y_norm[0], z_norm[0]],
                 [x_norm[1], y_norm[1], z_norm[1]],
                 [x_norm[2], y_norm[2], z_norm[2]]]
    cb_matrix = np.array(cb_matrix)
    print("cb_matrix: ", cb_matrix)
    cb_matrix_inv = np.linalg.inv(cb_matrix)
    S = np.array([cb_matrix_inv.dot(vect) for vect in S])
    return S

def get_error(I, I_new, dim, mask):
    I = np.reshape(I, (dim[0]*dim[1], dim[2]))
    I_new = np.reshape(I_new, (dim[0]*dim[1], dim[2]))
    Ierr = np.array([])
    for i in range(I.shape[1]):
        reconstructed = np.uint8(I_new[:,i])
        Ierri = np.array(np.double(I[:,i]) - reconstructed)/255
        if not Ierr.size:
            Ierr = Ierri**2
        else:
            Ierr += Ierri**2
    Ierr = np.sqrt(Ierr/ I.shape[1])
    Ierr = [element for i, element in enumerate(Ierr) if mask[i]]
    Ierr = np.array(Ierr)
    print("Evaluate scaled normal estimation by intensity error:")
    print("RMS: ", np.sqrt(np.mean(Ierr**2)))
    print("Mean: ", np.mean(Ierr))
    print("Median: ", np.median(Ierr))
    print("90 percentile: ", np.percentile(Ierr, 90))
    print("Max: ", np.max(Ierr))

def show_reconstructed():
    reconstructed = np.loadtxt("reconstructed.txt", delimiter=',')
    reconstructed = np.reshape(reconstructed, (200, 150))
    cv.imshow("reconstructed", reconstructed)
    cv.waitKey(0)
    cv.destroyAllWindows()
        
def show_txt():
    S = np.loadtxt("output.txt", delimiter=",")
    final_mask = np.loadtxt("final_mask.txt", delimiter=",")
    S = np.array(S).T
    #swap red and blue
    temp = S[0].copy()
    S[0] = S[2]
    S[2] = temp
    S = S.T
    S[:,1] = S[:,1].copy()
    #swap red and green but negate green
    temp = S[1].copy()
    S[1] = S[2]
    S[2] = temp
    S = np.reshape(S, (250, 188, 3))
#    compare_harvard_sn(final_mask)
    cv.imshow("Output", S)
    cv.waitKey(0)
    cv.destroyAllWindows()    

def main():
    # PARAMS
    height_scale = 0.4 #scale relative to 500
    mask = get_mask()
    parser = argparse.ArgumentParser(description="Perform unknown lighting photometric stereo on a dataset")
    parser.add_argument('-z', '--zloc', required=True,
                         help="Pixel coordinates in the image for the location where the surface normal is (0,0,1)")
    parser.add_argument('-s', '--savetxt', action='store_true', help="toggle this option if you want to store image into text file instead of displaying it.")
    parser.add_argument('-r', '--reconstruct', action='store_true', help="toggle this option if you want to reconstruct image into text file instead of displaying it.")

    args = parser.parse_args(sys.argv[1:])
    print("start")
    if args.reconstruct:
        show_reconstructed()
        sys.exit(0)
    if not args.savetxt:
        show_txt()
        sys.exit(0)
    dir_images = os.getcwd() + "/test_data/my_data/pisa"
    img_files = sorted([join(dir_images, f) for f in listdir(dir_images) if isfile(join(dir_images, f))])
    I = []
    # You need to find a pixel where the real surface norm is (0,0,1): This is 300, 225 in the harvard dataset: 200, 175 in adjusted
    z_coord = (int(args.zloc.split(",")[0]), int(args.zloc.split(",")[1]))
    z_coord = [1890, 752]
    y_coord = [1806, 752]
    x_coord = [1779, 1138]
    print("Z-cord: ", z_coord)
    for iterate in range(0, len(img_files)):
        file = img_files[iterate]
        print(file)
        img = cv.imread(file, 0)
        #img = cv.bitwise_or(img, img, mask=mask)
        if img.shape[0] > 500*height_scale:
            scale = img.shape[0]/(500*height_scale)
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
            mask = cv.resize(mask, dim, interpolation=cv.INTER_AREA)
            dim = img.shape
            if z_coord[0] > 500*height_scale or z_coord[1] > 500*height_scale:
                z_coord[0] = int(z_coord[0]/scale)
                z_coord[1] = int(z_coord[1]/ scale)
                y_coord[0] /= scale
                y_coord[1] /= scale
                x_coord[0] /= scale
                x_coord[1] /= scale
        I.append(img.flatten())
    print("y_coord in main,", y_coord)
    print("Scale factor: ", scale)
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
    print("s_hat shape ", s_hat.shape)
    print("l_hat shape ", l_hat.shape)
    # 6 points to the right of 180, 90
    ind_albedo = [int(1813/scale)*dim[1] + int(elem) for elem in np.linspace(int(357/scale), int(1137/scale), 6)]
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
    print("A shape: ", A.shape)
    S = np.matmul(s_hat, A)
    L = np.matmul(np.linalg.inv(A), l_hat)
    L = np.array([v/np.linalg.norm(v)*0.8 for v in L.T]).T

    print("S shape: ", S.shape)
    print("L shape: ", L.shape)
    
    S = change_of_basis(S, z_coord, y_coord, x_coord, dim)
    I_new = np.matmul(S, L)
    sample_reconstructed = np.reshape(I_new, (dim[0]*dim[1], dim[2])).T[0]
    mask = np.reshape(mask, (dim[0]*dim[1]))
    sample_reconstructed = np.array([element if mask[i] else 0 for i, element in enumerate(sample_reconstructed)])
    np.savetxt('reconstructed.txt', sample_reconstructed, delimiter=',')
    print("I_new shape", I_new.shape)
    get_error(I, I_new, dim, mask)

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
    # cv.imwrite("output.jpg", surface_normals)

    
if __name__ == "__main__":
    main()
