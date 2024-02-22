import numpy as np
import skimage as ski
from scipy.ndimage import convolve
from ImageFilter import *

def edge_thining(image, angle):
    M, N = image.shape
    Z = np.zeros((M, N))
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            if (angle[i, j] == 0) or (angle[i,j] == 180):
                q = image[i, j+1]
                r = image[i, j-1]

            elif (angle[i, j] == 45):
                q = image[i+1, j-1]
                r = image[i-1, j+1]

            elif (angle[i, j] == 90):
                q = image[i + 1, j]
                r = image[i - 1, j]

            elif (angle[i, j] == 135):
                q = image[i-1, j-1]
                r = image[i+1, j+1]

            #interpolation cases
            elif (angle[i, j] > 0 and angle[i, j] < 45):
                q = int((image[i, j+1] + image[i+1, j-1]) / 2)
                r = int((image[i, j-1] + image[i-1, j+1])/2)

            elif (angle[i, j] > 45 and angle[i, j] < 90):
                q = int((image[i+1, j-1] + image[i + 1, j]) / 2)
                r = int((image[i-1, j+1] + image[i - 1, j]) / 2)

            elif (angle[i, j] > 90 and angle[i, j] < 135):
                q = int((image[i + 1, j] + image[i-1, j-1]) / 2)
                r = int((image[i - 1, j] + image[i+1, j+1]) / 2)

            elif (angle[i, j] > 135 and angle[i, j] < 180):
                q = int((image[i-1, j-1] + image[i, j+1]) / 2)
                r = int((image[i+1, j+1] + image[i, j-1]) / 2)

            if (image[i, j] >= q) and (image[i, j] >= r):
                Z[i, j] = image[i, j]
            else:
                Z[i, j] = 0
    return Z

def sobelFilter(image):
    image = ski.color.rgb2gray(image)
    sobelfilterXDirection = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelfilterYDirection = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Gx = convolve(image, sobelfilterXDirection)
    Gy = convolve(image, sobelfilterYDirection)

    image = np.hypot(Gx, Gy)
    image = image / image.max() * 255
    theta = np.arctan2(Gy,Gx)

    return image, theta


def threshold(img, lowThresholdRatio=0.1, highThresholdRatio=0.2):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def cannyFilter(image, k0):
    image = apply_filter(image, k0, "hgtp")
    image, angle = sobelFilter(image)
    image = edge_thining(image, angle)

    image, weak, strong = threshold(image)

    print(weak, strong)
    #image = hysteresis(image, weak)

    return image

def createBoundingBox(self, filter="canny"):
    return None
