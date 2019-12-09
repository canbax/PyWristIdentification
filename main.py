from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.exposure import adjust_gamma
from skimage.measure import regionprops
from skimage.filters import gaussian, sobel_h, sobel_v, laplace
from skimage.color import rgb2gray, rgb2hsv, rgb2lab, rgb2xyz, rgb2ycbcr, rgb2yiq
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from math import ceil, atan
from scipy import ndimage

PIXEL_MAX_VAL = 255


def get_adjacency_matrix(segments: np.array):
    # slic returns labels from 0 to n
    n = segments.max() + 1
    r = np.zeros((n, n))

    x, y = segments.shape

    # horizontally iterate over pixels
    for i in range(x):
        for j in range(i + 1, y - 1):
            seg1 = segments[i][j]
            seg2 = segments[i][j + 1]
            if seg1 != seg2:
                r[seg1][seg2] = 1
                r[seg2][seg1] = 1

    # vertically iterate over pixels
    for j in range(y):
        for i in range(j + 1, x - 1):
            seg1 = segments[i][j]
            seg2 = segments[i + 1][j]
            if seg1 != seg2:
                r[seg1][seg2] = 1
                r[seg2][seg1] = 1

    return r


def get_differenceOfGaussians(img, sigma1, sigma2):
    filter1_size = 2 * ceil(3 * sigma1) + 1
    filter2_size = 2 * ceil(3 * sigma2) + 1

    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val) * 255

    return gaussian(img, sigma=sigma1) - gaussian(img, sigma=sigma2)


def get_7_gradient_maps(imgRGB):
    gamma = 0.4
    y = rgb2gray(adjust_gamma(imgRGB, gamma))
    gray = np.copy(y)
    y = get_differenceOfGaussians(y, 1, 2)
    # what is a?
    a = 0.1
    # why 3 ? why not 4?
    t = np.ones(y.shape) * 3
    # what is happening in below 3 lines?
    y = np.divide(y, np.mean(((abs(y)) ** a)) ** (1 / a))
    y = np.divide(y, np.mean((np.minimum(abs(y), t)) ** a) ** (1 / a))
    imgNorm = y * np.tanh(np.divide(y, t))

    imgLAB = rgb2lab(img)
    L = imgLAB[:, :, 0]
    A = imgLAB[:, :, 1]
    B = imgLAB[:, :, 2]
    # why 2.2?
    _lambda = 2.2
    S = np.sqrt((L - np.mean(L))**2 + (_lambda * (A - np.mean(A)))
                ** 2 + (_lambda * (B - np.mean(B)))**2)
    Smax = np.max(S)
    Smin = np.max(S)
    labNorm = (S - Smin) / (Smax - Smin)

    imgGx = sobel_h(gray)
    imgGy = sobel_v(gray)
    laplacian = laplace(gray)

    l1 = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]])
    l2 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])
    lapX = ndimage.convolve(gray, l1)
    lapY = ndimage.convolve(gray, l2)

    # what is going on below lines why are there hardcoded values like 4,5,6
    # lapY[:, 1:3] = lapY[:, [6, 5, 4]]
    # lapY[:, -2:] = lapY[:, [-3, -4 - 5]]
    # lapX[1:3, :] = lapX[[6, 5, 4], :]
    # lapX[-2:, :] = lapX[[-3, -4 - 5], :]
    # lapX = lapX[2:-1, 2:-1]
    # lapY = lapY[2:-1, 2:-1]

    # return labNorm, imgNorm, imgGx, imgGy, laplacian, lapX, lapY
    return np.array([labNorm, imgNorm, imgGx, imgGy, laplacian, lapX, lapY])


def extract_superpixel_features(stats, img, adj_matrix, binaryCropMask):

    imgHSV = rgb2hsv(img)
    imgLAB = rgb2lab(img)
    imgXYZ = rgb2xyz(img)
    imgYCbCr = rgb2ycbcr(img) / PIXEL_MAX_VAL
    imgYIQ = rgb2yiq(img)
    imgGray = rgb2gray(img)

    # set pixel range to [0,1]
    imgRGB = img / PIXEL_MAX_VAL
    imgRGBsum = np.sqrt(np.sum(np.square(imgRGB), axis=0))
    # broadcasting will make division in 3 channels
    imgRGBnorm = np.divide(imgRGB, imgRGBsum)

    gradient_maps = get_7_gradient_maps(imgRGB)
    minCoverArea = 0
    color_space_matrix = np.array(
        [imgRGB, imgRGBnorm, imgLAB, imgHSV, imgYCbCr, imgYIQ])

    feature_count = 450
    superpixel_features = np.zeros((len(stats), feature_count))
    for i, s in enumerate(stats):
        # what second condition means in matlab code?
        if s.area > 0:
            print('')

# superpix is 0 based label of superpixel
# stats is ordered by label (asc)


def get8neighbors4_superpixel(stats, adj_matrix, superpix):
    neighborhood = adj_matrix[superpix, superpix + 1:]
    neighbor_idxs = np.nonzero(neighborhood)

    up_neighbors = []
    up_cnt = 0
    down_neighbors = []
    down_cnt = 0
    right_neighbors = []
    right_cnt = 0
    left_neighbors = []
    left_cnt = 0

    slope_threshold = 0.5

    for nei in neighbor_idxs:
        x1, y1 = stats(superpix).centroid
        x2, y2 = stats(nei).centroid
        slope = abs((y2 - y1) / (x2 - x1))
        if slope > slope_threshold:
            if y2 > y1:
                up_neighbors.append({'label': nei, 'area': stats(nei).area})
                up_cnt = up_cnt + 1
            else:
                down_neighbors.append({'label': nei, 'area': stats(nei).area})
                down_cnt = down_cnt + 1
        else:
            if x2 > x1:
                right_neighbors.append({'label': nei, 'area': stats(nei).area})
                right_cnt = right_cnt + 1
            else:
                left_neighbors.append({'label': nei, 'area': stats(nei).area})
                left_cnt = left_cnt + 1

    up_neighbors = up_neighbors.sort(key=lambda x: x['area'])
    down_neighbors = down_neighbors.sort(key=lambda x: x['area'])
    right_neighbors = right_neighbors.sort(key=lambda x: x['area'])
    left_neighbors = left_neighbors.sort(key=lambda x: x['area'])
    
    
    
    
fname = 'img\\SEToriginalWristImages\\SET1\\0001_01_01_02_863_695_288_408_L.jpg'
num_superpixel = 200
img_height = 200
compactness = 8

img = plt.imread(fname)
img = resize(img, (img_height, img.shape[1]))
segments = slic(img, n_segments=num_superpixel, compactness=compactness)

# plt.imshow(mark_boundaries(img, segments))
# plt.show()

adj_matrix = get_adjacency_matrix(segments)
# region props ignore labels with 0 but slic labels are 0 indexed
stats = regionprops(segments + 1)

binaryCropMask = np.ones(img.shape)
extract_superpixel_features(stats, img, adj_matrix, binaryCropMask)

print(len(stats))
print(adj_matrix)
