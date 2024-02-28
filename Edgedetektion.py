import numpy as np
import skimage as ski
from scipy.ndimage import convolve
from ImageFilter import *
from scipy import signal

def edge_thining(img, angle):
    M, N = img.shape
    Z = np.zeros((M, N))
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
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
    if (len(image) == 3):
        image = ski.color.rgb2gray(image)

    sobelfilterXDirection = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelfilterYDirection = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Gx = convolve(image, sobelfilterXDirection)
    Gy = convolve(image, sobelfilterYDirection)

    image = np.hypot(Gx, Gy)
    image = image / image.max() * 255
    theta = np.arctan2(Gy, Gx)

    return image, theta

def scharrFilter(image):
    if (len(image) == 3):
        image = ski.color.rgb2gray(image)
        
    sobelfilterXDirection = np.array([[47, 0, -47], [162, 0, -162], [47, 0, -47]])
    sobelfilterYDirection = np.array([[47, 162, 47], [0, 0, 0], [-47, -162, -47]])

    Gx = convolve(image, sobelfilterXDirection)
    Gy = convolve(image, sobelfilterYDirection)

    image = np.hypot(Gx, Gy)
    image = image / image.max() * 255
    theta = np.arctan2(Gy, Gx)

    return image, theta

def threshold(img, low_threshold=0.1, high_threshold=0.2):
    low_threshold = 5
    high_threshold = 25

    M, N = img.shape

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)

    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    img[strong_i, strong_j] = strong
    img[weak_i, weak_j] = weak

    for i in range(1, M - 1):
        for j in range(1, N - 1):
           if (img[i, j] == weak):
                try:
                    if np.any(img[i-1:i+1, j-1:j+1] == strong):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def cannyFilter(image, k0, filter="sobel"):
    image = apply_filter(image, k0, "hgtp")
    if (filter == "scharr"):
        image, angle = scharrFilter(image)
    elif (filter == "sobel"):
        image, angle = sobelFilter(image)
    else:
        raise ValueError("filter is not avaiable")

    image = edge_thining(image, angle)

    #image = threshold(image)

    return image

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

def createBoundingBox(self, filter="canny"):
    return None
