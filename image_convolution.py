import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Numpy arrays

# Edge detection
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]])

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

edge_detection = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])

edge_detection2 = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])

# Blur
mean_filter = (1 / 9) * np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]])


mean_filter_11 = (1 / 11**2) * np.ones((11, 11))

# Separable Kernels -- run in sequence to produce same effect as one 33x33 kernel, but faster
mean_filter_33_x = (1 / 33) * np.ones((1, 33))
mean_filter_33_y = (1 / 33) * np.ones((33, 1))

# 33x33 for comparison
mean_filter_33_xy = (1 / (33*33)) * np.ones((33, 33))

gaussian_blur_3x3 = (1 / 16) * np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]])


gaussian_blur_5x5 = (1 / 256)  * np.array([[1, 4, 6, 4, 1],
                                         [4, 16, 24, 16, 4],
                                         [6, 24, 36, 24, 6],
                                         [4, 16, 24, 16, 4],
                                         [1, 4, 6, 4, 1]])

gaussian_blur_9x9 = np.array([[0.000814, 0.001918, 0.003538, 0.005108, 0.005774, 0.005108, 0.003538, 0.001918, 0.000814],
                            [0.001918, 0.00452, 0.008338, 0.012038, 0.013605, 0.012038, 0.008338, 0.004552, 0.001918],
                            [0.003538, 0.008338, 0.015378, 0.022203, 0.025094, 0.022203, 0.015378, 0.008338, 0.003538],
                            [0.005108, 0.012038, 0.022203, 0.032057, 0.036231, 0.032057, 0.022203, 0.012038, 0.005108],
                            [0.005774, 0.013605, 0.025094, 0.036231, 0.04095, 0.036231, 0.025094, 0.013605, 0.005774],
                            [0.005108, 0.012038, 0.022203, 0.032057, 0.036231, 0.032057, 0.022203, 0.012038, 0.005108],
                            [0.003538, 0.008338, 0.015378, 0.022203, 0.025094, 0.022203, 0.015378, 0.008338, 0.003538],
                            [0.001918, 0.00452, 0.008338, 0.012038, 0.013605, 0.012038, 0.008338, 0.004552, 0.001918],
                            [0.000814, 0.001918, 0.003538, 0.005108, 0.005774, 0.005108, 0.003538, 0.001918, 0.000814]])

# Sharpen
sharpen = np.array([[1, -2, 1],
                    [-2, 5, -2],
                    [1, -2, 1]])

sharpen2 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

sharpen3 = np.array([[-1, -2, -1],
                     [-2, 13, -2],
                     [-1, -2, -1]])


def compare_images(img1, img2):
    fig, (orig_img, new_img) = plt.subplots(1, 2, figsize=(15, 6))
    
    orig_img.imshow(img1)
    orig_img.set_title('Original Image')
    orig_img.set_axis_off()

    new_img.imshow(img2)
    new_img.set_title('Modified Image')
    new_img.set_axis_off()
    plt.show()

# *args used for arbitrary number of arguments ("args" is convention, but *kernels etc. would work too)
def convolve(img, *args):

    # Move the color channels to index 0, to make looping over them easier
    img = img.transpose(2,0,1).astype(np.float32)
    num_kernels = len(args)

    channels, height, width = img.shape # Tuple unpacking
    
    # This array stores any intermediate steps in the convolution -- when there are multiple kernels
    outputs = [np.zeros((channels, height, width)) for i in range(num_kernels + 1)] # List comprehension
    outputs[0] = img

    # Do the convolution
    for i in range(1, num_kernels + 1):
        for j in range(channels):
            outputs[i][j] = signal.convolve2d(outputs[i-1][j], args[i-1], boundary='symm', mode='same')

    # Reset the pixel values to 0-255, and return the array to its original shape
    result = outputs[num_kernels] * 255
    result = np.absolute(result.transpose(1,2,0)).astype(np.uint8)
    
    return result


# Load the image
img = mpimg.imread("Stream.png")

# Can do multiple kernels in one function call, just keep adding arguments
new_img = convolve(img, mean_filter_33_x, mean_filter_33_y)

# See the result
compare_images(img, new_img)

