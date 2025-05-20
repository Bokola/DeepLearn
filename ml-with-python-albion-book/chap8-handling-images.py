from os import mkdir

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow_datasets.datasets.radon.radon_dataset_builder import features
import torch
from torchvision import transforms
import torchvision.models as models
import wget
import urllib.request



plt.ion()
matplotlib.use('TkAgg')

# https://stackoverflow.com/questions/61583476/pycharm-not-displaying-matplotlib-plot

# 8.1 Loading Images

# load image
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# show image
# plt.imshow(image, cmap="gray"), plt.axis("off")
# plt.show()

# images are nd arrays
image

# load image in colour
image_gbr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# show pixel
image_gbr[0,0]

# convert to RGB
image_rgb = cv2.cvtColor(image_gbr, cv2.COLOR_BGR2RGB)

# 8.2 Saving images

# load image
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# save image
cv2.imwrite("images/plane_gray.jpg", image)

# 8.3 Resizing images

image_50x50 = cv2.resize(image, (50, 50))
# 8.4 Cropping images

# select first half of columns and all rows
image_cropped = image[:,:128]

# 8.5 Blurring images
# uses kernels
image_blurred = cv2.blur(image, (5,5))


# 8.6 Sharpening images
# - create a kernel that highlights target pixel then apply
# it to image using filter2D

# create kernel
kernel = np.array([[0,-1,0],[-1,5,-1], [0,-1,0]])
# sharpen image
image_sharp = cv2.filter2D(image, -1, kernel)

# 8.7 Enhancing contrast
# uses histogram equalization
image_enhanced = cv2.equalizeHist(image)
# Show image
plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()

# for colored images you first need to convert the image to
# the YUV color format: the Y is the luma/brightness, U and V
# denote the color
image_bgr = cv2.imread("images/plane.jpg")

# convert to YUV
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
# apply histogram equalization

image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
# convert to RGB
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
# show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# 8.8 Isolating colors

# load image
image_bgr = cv2.imread('images/plane_256x256.jpg')
# convert BGR to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
# define range of blue values in HSV
lower_blue = np.array([50,100,50])
upper_blue = np.array([130,255,255])
# create a mask
image_bgr_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
# convert BGR to RGB
image_rgb = cv2.cvtColor(image_bgr_mask, cv2.COLOR_BGR2RGB)
# show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# 8.9 Binarizing images: convert from gray scale to black/white
# output a simplified version of an image

# Load image as grayscale
image_grey = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# apply adaptive thresholding
max_output_value = 255; neighborhood_size=99; subtract_from_mean=10
image_binarized = cv2.adaptiveThreshold(image_grey,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean)

# show image
plt.imshow(image_binarized, cmap="gray"), plt.axis("off")

# 8.10 Removing backgrounds

# Load image and convert to RGB
image_bgr = cv2.imread('images/plane_256x256.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Rectangle values: start x, start y, width, height
rectangle = (0, 56, 256, 150)
# Create initial mask
mask = np.zeros(image_rgb.shape[:2], np.uint8)
# Create temporary arrays used by grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
# Run grabCut
cv2.grabCut(image_rgb, # Our image
mask, # The Mask
rectangle, # Our rectangle
bgdModel, # Temporary array for background
fgdModel, # Temporary array for background
5, # Number of iterations
cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle
# Create mask where sure and likely backgrounds set to 0, otherwise 1
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
# Multiply image with new mask to subtract background
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
# Show image
plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()

# 8.11 Detecting edges
# Load image as grayscale
image_gray = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
# Calculate median intensity
median_intensity = np.median(image_gray)
# Set thresholds to be one standard deviation above and below median intensity
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
# Apply Canny edge detector
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
# Show image
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()

# 8.12 Detecting corners

# Load image
image_bgr = cv2.imread("images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)
# Set corner detector parameters
block_size = 2
aperture = 29
free_parameter = 0.04
# Detect corners
detector_responses = cv2.cornerHarris(image_gray,
block_size,
aperture,
free_parameter)
# Large corner markers
detector_responses = cv2.dilate(detector_responses, None)
# Only keep detector responses greater than threshold, mark as white
threshold = 0.02
image_bgr[detector_responses >
threshold *
detector_responses.max()] = [255,255,255]
# Convert to grayscale
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# Show image
plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()

# 8.13 Creating features for machine learning
# - use Numpy's flatten to convert multidimensional array to vectors

# Load image as grayscale
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# resize image to 10 pixels by 10 pixels
image_10x10 = cv2.resize(image, (10,10))
# convert image data to one-dimensional vector
image_10x10.flatten()
# images are represented as grids of pixels; pixel intensity of 1 if white, 0 if black
plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
plt.show()

# 8.14 Encoding color histograms as features

np.random.seed(0)

# load image
image_bgr = cv2.imread('images/plane_256x256.jpg', cv2.IMREAD_COLOR)
# convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# create a list for feature values
features = []
# calculate the histogram for each color channel
colors = ('r', 'g', 'b')

for i, c in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # image
                             [i], # index of channel
                             None, # no mask
                             [256], # histogram size
                             [0, 256] # range
                             )
    plt.plot(histogram, color = c)
    plt.xlim([0, 256])
    features.extend(histogram)

# create a vector for an observation's feature values
observation = np.array(features).flatten()
# show observations's value for first 5 features
observation[0:5]

# 8.15 Using pretrained embeddings as features

# Load image
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)

# convert to pytorch data type
convert_tensor = transforms.ToTensor()
pytorch_image = convert_tensor(np.array(image_rgb))

# load the pretrained model
model = models.resnet18(pretrained=True)
# select specific layer of model we want to output from
layer = model._modules.get("avgpool")
# set model to evaluate mode
model.eval()
# infer embedding with no_grad option
with torch.no_grad():
    embedding = model(pytorch_image.unsqueeze(0))

print(embedding.shape)

