import numpy as np
import skimage as ski
from scipy.ndimage import convolve
from ImageFilter import *
from scipy import signal
import collections

def edge_thining(img, angle):
    M, N = img.shape
    Z = np.zeros((M, N))
    angle[angle < 0] += np.pi

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < np.pi/8) or (np.pi*(7/8) <= angle[i, j] <= np.pi):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 135
                elif (np.pi*(5/8) <= angle[i, j] < np.pi*(7/8)):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (np.pi*(3/8) <= angle[i, j] < np.pi*(5/8)):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 45
                elif (np.pi/8 <= angle[i, j] <np.pi*(3/8)):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z

def sobelFilter(image):
    if (len(image.shape) == 3):
        image = ski.color.rgb2gray(image)

    sobelfilterXDirection = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelfilterYDirection = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Gx = convolve(image, sobelfilterXDirection)
    Gy = convolve(image, sobelfilterYDirection)

    magnitude = Gx*Gx
    magnitude += Gy*Gy

    #image = np.hypot(Gx, Gy)
    image = np.sqrt(magnitude)
    image = image / image.max() * 255
    theta = np.arctan2(Gy, Gx)

    return image, theta

def scharrFilter(image):
    if (len(image.shape) == 3):
        image = ski.color.rgb2gray(image)

    scharrfilterXDirection = np.array([[47, 0, -47], [162, 0, -162], [47, 0, -47]])
    scharrfilterYDirection = np.array([[47, 162, 47], [0, 0, 0], [-47, -162, -47]])

    Gx = convolve(image, scharrfilterXDirection)
    Gy = convolve(image, scharrfilterYDirection)

    image = np.hypot(Gx, Gy)
    image = image / image.max() * 255
    theta = np.arctan2(Gy, Gx)

    return image, theta

def threshold(img):
    low_threshold = 5
    high_threshold = 25
    M, N = img.shape

    Z = np.zeros((M, N))

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)

    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    Z[strong_i, strong_j] = strong
    Z[weak_i, weak_j] = weak
    Z[zeros_i, zeros_j] = 0

    for i in range(len(weak_i)):
            try:
                if np.any(Z[weak_i[i]-1:weak_i[i]+2, weak_j[i]-1:weak_j[i]+2] == strong):
                    Z[weak_i[i], weak_j[i]] = strong
                else:
                    Z[weak_i[i], weak_j[i]] = 0
            except IndexError as e:
                pass
    return Z

def cannyFilter(image,sigma=1, filterImage="sobel"):
    image1 = ski.filters.gaussian(image, sigma)
    if (filterImage == "scharr"):
        image2, angle = scharrFilter(image)
    elif (filterImage == "sobel"):
        image2, angle = sobelFilter(image1)
    else:
        raise ValueError("filter is not avaiable")

    image3 = edge_thining(image2, angle)

    image4 = threshold(image3)

    return image1, image2, image3, image4

def createBoundingBox(image, sigma=1, filterImage2="sobel"):
    image_type = True
    if len(image.shape) != 3:
        image_type = False

    image2 = cannyFilter(image, sigma, filterImage2)[3]

    white_pixleX, white_pixelY = np.where(image2 == 255)

    x_min, x_max = white_pixleX.min(), white_pixleX.max()
    y_min, y_max = white_pixelY.min(), white_pixelY.max()

    image3 = image.copy()

    if image_type:
        for i in range(y_min, y_max):
            image3[x_min, i, :] = 0
            image3[x_max, i, :] = 0

        for j in range(x_min, x_max):
            image3[j, y_min, :] = 0
            image3[j, y_max, :] = 0

    else:
        for i in range(y_min, y_max):
            image3[x_min][i] = 255
            image3[x_max][i] = 255

        for j in range(x_min, x_max):
            image3[j][y_min] = 255
            image3[j][y_max] = 255

    return image3

def otsu_threshold(image):
    if len(image.shape) == 3:
        image1 = ski.color.rgb2gray(image.copy())
    else:
        image1 = image.copy().reshape(-1, )

    countsPerPixel = np.bincount(image1.ravel())

    maxPix = len(countsPerPixel)
    M, N = image.shape
    image_size = M * N
    interClassVar = []

    for t in range(maxPix):  # Iterate over possible thresholds
        omega_0 = np.sum(countsPerPixel[:t]) / image_size
        omega_1 = np.sum(countsPerPixel[t:]) / image_size

        if omega_0 == 0 or omega_1 == 0:
            continue

        phi_0 = np.sum(np.arange(t) * countsPerPixel[:t]) / (omega_0 * image_size)
        phi_1 = np.sum(np.arange(t, maxPix) * countsPerPixel[t:]) / (omega_1 * image_size)

        interClassVar.append((omega_0 * omega_1) * ((phi_0 - phi_1) ** 2))

    threshold = np.argmax(interClassVar)

    return_image = np.where(image > threshold, 255, 0)

    return return_image, threshold, interClassVar

def detect_line_segments(input_image, edge_pixels, vote_threshold, segment_length_criteria):
    output_segments = []

    while edge_pixels:
        best_pixel = find_best_distinguished_pixel(edge_pixels)  # Function to find the best-distinguished pixel

        paired_pixel = None
        theta = calculate_line_parameters(best_pixel, paired_pixel)  # Function to calculate line parameters
        votes = calculate_votes(theta, edge_pixels)  # Function to calculate votes using voting kernel

        max_vote_index = np.argmax(votes)
        max_vote = votes[max_vote_index]

        if max_vote < vote_threshold:
            continue

        if meets_criteria(theta, segment_length_criteria):  # Function to check if parameters meet criteria
            output_segments.append(theta)

        # Remove pixels on the primitive from input image
        edge_pixels = remove_pixels(edge_pixels, theta)  # Function to remove pixels on the primitive

    return output_segments

# Define functions used in the algorithm
def find_best_distinguished_pixel(img, edge_pixels):
    best_distinguished_pixel_x = []
    best_distinguished_pixel_y = []
    for i in range(len(edge_pixels[0])):
        count = 0
        for k in range(edge_pixels[0][i] - 1, edge_pixels[0][i] + 1):
            for l in range(edge_pixels[1][i] - 1, edge_pixels[1][i] + 1):
                if (img[k, l] == 255): #and (k == edge_pixels[0][i] and l == edge_pixels[1][i]) == False):
                        count += 1
        if (count == 1):
            best_distinguished_pixel_x.append(edge_pixels[0][i])
            best_distinguished_pixel_y.append(edge_pixels[1][i])

    return np.array([best_distinguished_pixel_x, best_distinguished_pixel_y])

def find_another_edge_pixel(edge_pixels):
    # Implementation specific to your needs
    pass

def calculate_line_parameters(pixel1, pixel2):
    # Implementation specific to your needs
    pass

def calculate_votes(theta, edge_pixels):
    # Implementation specific to your needs
    pass

def meets_criteria(theta, segment_length_criteria):
    # Implementation specific to your needs
    pass

def remove_pixels(edge_pixels, theta):
    # Implementation specific to your needs
    pass

