# A failed Attempt to implement SURF Algorithm
# By Hadjerci Mohammed Allaeddine
# M2 MIV


#TODO: Optimize the code
#TODO: Rotation invariant

# Average Execution Time for now is 50 seconde


# Resources
# https://medium.com/software-incubator/introduction-to-surf-speeded-up-robust-features-c7396d6e7c4e
# https://www.vision.ee.ethz.ch/~surf/eccv06.pdf
# https://medium.com/lis-computer-vision-blogs/scale-invariant-feature-transform-sift-detector-and-descriptor-14165624a11
# Face recognition using SURF features Geng Du*, Fei Su, Anni Cai
# https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_00517.pdf


import cv2 
import numpy as np
import math
import time
from scipy import ndimage


# integral images or summed area table
# Integral(y, x) = sum(  sum( I(i, j)  ) )
#                  i<=y  j<=x
# used to reduce time complexity in box type convolution filters
def getIntegralImg(img):
    h, w = img.shape
    integral = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            integral[y, x] = np.sum(img[:y+1, :x+1])
    
    return integral



# https://www.vision.ee.ethz.ch/~surf/eccv06.pdf
# Hessian matrix-based interest points
# Hessian matrix have good performance
# Surf relies on the determinant of the Hessian matrix
# To adapt to any scale the given image is filtred by
# a Gaussian kernel
# for given Point X = (x, y) the Hessian matrix is defined by
# H(x, sigma) in x at scale sigma
# H(x, sigma) = [[Lxx(x, sigma), Lxy(x, sigma)],
#                [Lxy(x, sigma), Lyy(x, sigma)]]
# Lxx(x, sigma) is the convolution of the gaussian second order derivative with image I in point x

# in order to calculate the determinant of the hessian matrix
# we need to convolve with gaussian kernel
# then second order derivitive

# gaussian second order derivative used in Lxx(X, sigma)
# is given with ( d ^2 / dx^2 ) * g(sigma)
# g(sigma) = 1/(2 pi sigma) * exponent(-(x^2 + y^2) / 2 sigma^2)

# approximation of gaussian partial derivative with a box filter
# was proposed by Herbert
# det(H approx) = Dxx*Dyy - (0.9*Dxy)^2
# Dxx, Dyy, Dxy represent the convolution of box filters with the image
# sigma = 1.2 magic number
# we need to Approximate the seconde order gaussian derivative
# to do that we use box filter

#gauss second derivative approximation from research papers
def gauss_approxiamtion(size=9):

    Dxx = np.zeros((size, size), dtype=np.float)
    Dyy = np.zeros((size, size), dtype=np.float)
    Dxy = np.zeros((size, size), dtype=np.float)

    offsetyy = int(size / 3)

    Dxx[2:-2, :offsetyy] = 1
    Dxx[2:-2, offsetyy:offsetyy*2] = -2
    Dxx[2:-2, offsetyy*2:] = 1 
    Dxx_rect = [[ (2, 0), (2, offsetyy-1), (-2, offsetyy-1), (-2, 0)]
               ,[ (2, offsetyy), (2, offsetyy*2-1), (-2, offsetyy*2-1), (-2,offsetyy)]
               ,[ (2, offsetyy*2), (2, size-1), (-2, size-1), (-2, offsetyy*2)]]


    Dyy[:offsetyy, 2:-2] = 1
    Dyy[offsetyy:offsetyy*2, 2:-2] = -2
    Dyy[offsetyy*2:, 2:-2] = 1 
    Dyy_rect = [[ (0, 2), (0, -2), (offsetyy-1, -2), (offsetyy-1, 2)]
               ,[ (offsetyy, 2), (offsetyy, -2), (offsetyy*2-1, -2), (offsetyy*2-1, 2)]
               ,[ (offsetyy*2, 2), (offsetyy*2, -2), (size-1, -2), (size-1, 2)]]
    

    Dxy[1:offsetyy+1, 1:offsetyy+1] = 1
    Dxy[1:offsetyy+1, offsetyy*2-1:-1] = -1
    Dxy[offsetyy*2-1:-1, 1:offsetyy+1] = -1
    Dxy[offsetyy*2-1:-1, offsetyy*2-1:-1] = 1

    Dxy_rect = [[ (1, 1), (1, offsetyy), (offsetyy, offsetyy), (offsetyy, 1)]
               ,[ (1, offsetyy*2-1), (1, -1), (offsetyy, -1), (offsetyy, offsetyy*2-1)]
               ,[ (offsetyy*2-1, 1), (offsetyy*2-1, offsetyy), (-1, offsetyy), (-1, 1)]
               ,[ (offsetyy*2-1, offsetyy*2-1), (offsetyy*2-1, -1), (-1, -1), (-1, offsetyy*2-1)]]


    return [(Dxx, Dxx_rect), (Dyy, Dyy_rect), (Dxy, Dxy_rect)]


# from SURF Algorithm Implementation on FPGA publication
def L(image, D):
    iH, iW = image.shape
    kH, kW = D[0].shape
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    m_D, coords = D

    for y in np.arange(pad, iH + pad):

        for x in np.arange(pad, iW + pad):

            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            k = (roi * m_D).sum()
            '''for coord in coords:
                I0 = roi.item(coord[0][0], coord[0][1])
                D0 = m_D.item(coord[0][0], coord[0][1])

                I1 = roi.item(coord[2][0], coord[2][1])
                D1 = m_D.item(coord[2][0], coord[2][1])

                I2 = roi.item(coord[1][0], coord[1][1])
                D2 = m_D.item(coord[1][0], coord[1][1])

                I3 = roi.item(coord[3][0], coord[3][1])
                D3 = m_D.item(coord[3][0], coord[3][1])

                k += (I0 * D0) + (I1 * D1) - (I2 * D2) - (D3 * I3)'''
            
            output.itemset((y - pad, x - pad), k)


    # rescale the output image to be in the range [0, 255]
    # output_new = rescale_intensity(output, in_range=(0, 255))
    # output_new = (output_new * 255).astype("uint8")

    # return the output image
    return output


# hessian approximated determinant
def hdet(Lxx, Lyy, Lxy):
    D = (Lxx * Lyy) - 0.9**2 * Lxy**2
    D /= np.linalg.norm(D)
    return D


# WHAT TO DO, WHAT TO DO
# get Octaves
# let's start with 3 octaves
# https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_00517.pdf
def genOctaves(integral_img, nbOctaves):
    octaves = []
    first_kernel_size = 9
    next_octave = 15
    current_octave = 9
    for i in range(nbOctaves):
        octave = []
        for j in range(4):
            Dxx, Dyy, Dxy = gauss_approxiamtion(current_octave)
            Lxx = L(integral_img, Dxx)
            Lyy = L(integral_img, Dyy)
            Lxy = L(integral_img, Dxy)
            octaves.append(hdet(Lxx, Lyy, Lxy))
            octave.append(current_octave)
            current_octave, next_octave = next_octave, first_kernel_size + next_octave - 3

        print('Octave ', i+1, ' Calculated')

        first_kernel_size = octave[1]
        next_octave = octave[-1]
        current_octave = first_kernel_size
    
    return octaves


# difference of gaussian to make octaves of size 3 
# so we can apply non-maximum suppression in a 3 × 3 × 3 neighbourhood
def DoG(octaves):
    Do = []
    for i in range(1, 4):
        dif = octaves[i-1] - octaves[i]
        Do.append(dif)

    return Do


# non-maximum suppression
# https://homes.esat.kuleuven.be/~konijn/publications/2006/eth_biwi_00446.pdf
def nms2d(img, n=1):
    h, w = img.shape
    range1 = set(range(0, h - n))
    range2 = set(range(0, w - n))
    range3 = set(range(n-1, 2*n + 1))

    rangey = range1.intersection(range3)
    rangex = range2.intersection(range3)

    candidates = []

    for i in rangey:
        range33 = set(range(i, i + n))
        for j in rangex:
            mi, mj = i, j
            for i2 in range(i, i + n):
                for j2 in range(j, j + n):
                    if(img.item(i2, j2) > img.item(mi, mj)):
                        mi, mj = i2, j2
            
            
            range11 = set(range(mi - n, mi + n))
            range22 = set(range(mj - n, mj + n))
            range44 = set(range(j, j + n))

            rangeyy = range11.intersection(range33)
            rangexx = range22.intersection(range44)

            for i2 in rangeyy:
                for j2 in rangexx:
                    if(img.item(i2, j2) > img.item(mi, mj)):
                        return candidates

            candidates.append((mi, mj))
    
    return candidates


# extrema localization intrest point
def ex_intrest(intrest, compare1, compare2, threshold):
    h, w = intrest.shape
    candidates = []

    for y in range(1, h-1):
        for x in range(1, h-1):
            roi = np.array([intrest[y-1:y+1, x-1:x+1], compare1[y-1:y+1, x-1:x+1], compare2[y-1:y+1, x-1:x+1]]).flatten()
            if np.max(roi) == intrest[y, x] and intrest[y, x] > threshold:
                candidates.append((y, x))
    
    return candidates


# Interpolation
# ??????

# Trying to implement Rotation Invarient using FAILED
# Haar Wavelet

# Haar Wavelet Filters
horizontal_wavelet = np.zeros((7, 7))
horizontal_wavelet[:, :3] = -1
horizontal_wavelet[:, 3:] = 1
horizontal_wavelet = [horizontal_wavelet, [ [(0, 0), (0, 2), (-1, 2), (0, -1)],
                                            [(0, 3), (0, -1), (-1, -1), (-1, 3)] ]]

vertical_wavelet = np.zeros((7, 7))
vertical_wavelet[:3, :] = -1
vertical_wavelet[3:, :] = 1
vertical_wavelet = [vertical_wavelet, [ [(0, 0), (0, -1), (2, 0), (2, -1)],
                                            [(3, 0), (3, -1), (-1, -1), (-1, 0)] ]]

##################################################-MAIN SECTION-#################################################################

img = cv2.imread('lenna.jpg', 0)
#img_cropped = img[100:300, 100:300].copy()
img_cropped = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)

img_int = img.copy()
img_int_cropped = img_cropped.copy()

# Orientation Not Working
#img_cropped = ndimage.rotate(img_cropped, 90)

#img_int = getIntegralImg(img)
#img_int_cropped = getIntegralImg(img_cropped)

init_interest_point = []
init_intrr = []

n = 3

start = time.time()

#scale
s = 0

octaves = genOctaves(img_int, 1)
octaves_dog = DoG(octaves)
'''init_interest_point.append([nms2d(octaves_dog[0], n), 1.2*9*2/9])
init_interest_point.append([nms2d(octaves_dog[1], n), 1.2*15*2/9])
init_interest_point.append([nms2d(octaves_dog[2], n), 1.2*21*2/9])'''
init_interest_point.append([ex_intrest(octaves_dog[1], octaves_dog[0], octaves_dog[2], 0.001), 1.2*9*2/9])


new_octaves = genOctaves(img_int_cropped, 1)
new_octaves_dog = DoG(new_octaves)
'''init_intrr.append([nms2d(new_octaves_dog[0], n), 1.2*9*2/9])
init_intrr.append([nms2d(new_octaves_dog[1], n), 1.2*15*2/9])
init_intrr.append([nms2d(new_octaves_dog[2], n), 1.2*21*2/9])'''
init_intrr.append([ex_intrest(new_octaves_dog[1], new_octaves_dog[0], new_octaves_dog[2], 0.001), 1.2*9*2/9])

end = time.time()

for cords, s in init_interest_point:
    for cord in cords:
        img[cord] = 0

for cords, s in init_intrr:
    for cord in cords:
        img_cropped[cord] = 0

cv2.imshow('original', img)
cv2.imshow('cropped', img_cropped)

print('Time Elapsed : ', end-start)

cv2.waitKey(0)
cv2.destroyAllWindows()
