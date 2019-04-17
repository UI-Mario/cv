
import string, sys
import os.path
import argparse
import numpy
import math
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve
import cv2
from cv2 import imread, imwrite

class CannyEdgeDetector:
    def __init__(self):

        return

    def gaussian(self, image, sigma):
        return gaussian_filter(image, sigma)

    def gradients(self, image):
        sobel_kernel_x = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], numpy.int32)
        sobel_kernel_y = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], numpy.int32)

        g_x = convolve(image, sobel_kernel_x)
        g_y = convolve(image, sobel_kernel_y)

        # 等价于sqrt(x² + y²)
        g = numpy.hypot(g_x, g_y)

        # 方向
        theta = numpy.arctan2(g_y, g_x)

        return g, theta

    # 非极大值抑制
    def suppression(self, image, theta):
        y_length, x_length = image.shape

        s = numpy.zeros((y_length, x_length), dtype=numpy.int32)

        bound_index = lambda i, max_i: max(min(i, max_i), 0)

        for y in range(y_length):
            for x in range(x_length):
                # 
                pixel_index = int(math.floor((theta[y, x]%math.pi)/(math.pi/4.0)))

                x_index = 1 # if(pixel_index == 0)
                y_index = 0 # if(pixel_index == 0)
                if pixel_index == 1:
                    x_index = 1
                    y_index = 1
                elif pixel_index == 2:
                    x_index = 0
                    y_index = 1
                elif pixel_index > 2:
                    x_index = -1
                    y_index = 1

                image_yx = image[y, x]

                if (image_yx >= image[bound_index(y + y_index, y_length-1), bound_index(x + x_index, x_length-1)] and image_yx >= image[bound_index(y - y_index, y_length-1), bound_index(x - x_index, x_length-1)]):
                    s[y, x] = image_yx
        return s


    def threshold(self, image, lower_t, upper_t):
        none_x, none_y = numpy.where(image < lower_t)
        weak_x, weak_y = numpy.where((image >= lower_t) & (image <= upper_t))
        strong_x, strong_y = numpy.where(image > upper_t)

        image[none_x, none_y] = numpy.int32(0)
        image[weak_x, weak_y] = numpy.int32(50)
        image[strong_x, strong_y] = numpy.int32(255)
        return image


    def hysteresis(self, image):
        bound_index = lambda i, max_i: max(min(i, max_i), 0)

        y_length, x_length = image.shape
        for y in range(y_length):
            for x in range(x_length):
                if image[y, x] == numpy.int32(50):
                    for y_t in [-1, 0, 1]:
                        for x_t in [-1, 0, 1]:
                            if (image[bound_index(y + y_t, y_length-1), bound_index(x + x_t, x_length-1)] == numpy.int32(255)):
                                image[y, x] = numpy.int32(255)
                                break
                    if image[y, x] == numpy.int32(50):
                        image[y, x] = numpy.int32(0)
        return image

    def execute(self, image, sigma, lower_t, upper_t):
        image = self.gaussian(image, sigma)
        image, theta = self.gradients(image)
        image = self.suppression(image, theta)
        image = self.threshold(image, lower_t, upper_t)
        image = self.hysteresis(image)

        return image

# main (DRIVER)
def main():

    outputFileName = "output.jpg"
    input_filename = '1.jpg'

    image = imread(input_filename, cv2.IMREAD_GRAYSCALE).astype("int32")
    # image = cv2.resize(image, (312,416))
    # TODO: could these be passed-in or progrmaticly determined?
    sigma = 1
    lower_t = 10
    upper_t = 50

    image = CannyEdgeDetector().execute(image, sigma, lower_t, upper_t)
    imwrite(outputFileName, image)
    # 
    return 0

if __name__ == "__main__":
    main()

