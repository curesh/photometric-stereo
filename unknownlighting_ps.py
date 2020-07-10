import cv2 as cv
import numpy as np

def main():
    print("start")
    dir_images = ""
    I = []
    for file in dir_images:
        orig_img = cv.imread(dir_image, 0)
        I.append(orig_img)
        
    # Might have to tranpose first before svd
    u, sigma, vh = np.linalg.svd(I)
    u_clean = u[:,0:3]
    sigma_clean = s[0:3]
    vh_clean = vh[0:3]
    s_hat = np.linalg.matmul(u_clean, np.sqrt(sigma_clean))
    l_hat = np.linalg.matmul(np.sqrt(sigma_clean), vh_clean)

if __name__ == "__main__":
    main()