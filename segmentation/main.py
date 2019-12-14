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
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from os import listdir
import time
from random import randint

PIXEL_MAX_VAL = 255


def get_adjacency_matrix(segments: np.array):
    # slic returns labels from 0 to n
    n = segments.max() + 1
    r = np.zeros((n, n))

    x, y = segments.shape

    # horizontally iterate over pixels
    for i in range(x):
        for j in range(0, y - 1):
            seg1 = segments[i][j]
            seg2 = segments[i][j + 1]
            if seg1 != seg2:
                r[seg1][seg2] = 1
                r[seg2][seg1] = 1

    # vertically iterate over pixels
    for j in range(y):
        for i in range(0, x - 1):
            seg1 = segments[i][j]
            seg2 = segments[i + 1][j]
            if seg1 != seg2:
                r[seg1][seg2] = 1
                r[seg2][seg1] = 1

    return r


def get_differenceOfGaussians(img, sigma1, sigma2):
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

    imgLAB = rgb2lab(imgRGB)
    L = imgLAB[:, :, 0]
    A = imgLAB[:, :, 1]
    B = imgLAB[:, :, 2]
    # why 2.2?
    _lambda = 2.2
    S = np.sqrt((L - np.mean(L))**2 + (_lambda * (A - np.mean(A)))
                ** 2 + (_lambda * (B - np.mean(B)))**2)
    Smax = np.max(S)
    Smin = np.min(S)
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
    imgYCbCr = rgb2ycbcr(img) / PIXEL_MAX_VAL
    imgYIQ = rgb2yiq(img)

    # set pixel range to [0,1]
    imgRGB = img / PIXEL_MAX_VAL
    imgRGBsum = np.sqrt(np.sum(np.square(imgRGB), axis=0))
    # broadcasting will make division in 3 channels
    imgRGBnorm = np.divide(imgRGB, imgRGBsum)

    gradient_maps = get_7_gradient_maps(imgRGB)
    minCoverPercentage = 0.5
    color_spaces = [imgRGB, imgRGBnorm, imgLAB, imgHSV, imgYCbCr, imgYIQ]

    features = []
    for i, s in enumerate(stats):
        # get features of superpixels inside a mask
        if s.area <= 0 or np.mean(binaryCropMask[s.coords[:, 0], s.coords[:, 1]]) <= minCoverPercentage:
            continue
        feature = []
        # put coords of superpixel as a feature
        feature.append(s.centroid[0])
        feature.append(s.centroid[1])

        # get features of 8 neighbors and itself
        neighbors = get8neighbors4_superpixel(stats, adj_matrix, i)
        neighbors.append(i)
        for color_space in color_spaces:
            for nei in neighbors:
                for channel in range(3):
                    coords = stats[nei].coords
                    color_values = color_space[coords[:,
                                                      0], coords[:, 1], channel]
                    feature.append(np.mean(color_values))
                    feature.append(np.var(color_values))

        for gradient_map in gradient_maps:
            for nei in neighbors:
                coords = stats[nei].coords
                values = gradient_map[coords[:, 0], coords[:, 1]]
                feature.append(np.mean(values))
                feature.append(np.var(values))
        features.append(feature)

    return features

# superpix is 0 based label of superpixel
# stats is ordered by label (asc)


def get8neighbors4_superpixel(stats, adj_matrix, superpix):
    neighborhood = adj_matrix[superpix, :]
    neighbor_idxs = np.nonzero(neighborhood)[0]

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
        x1, y1 = stats[superpix].centroid
        x2, y2 = stats[nei].centroid
        slope = abs((y2 - y1) / (x2 - x1 + 1e-5))
        if slope > slope_threshold:
            if y2 > y1:
                up_neighbors.append({'idx': nei, 'area': stats[nei].area})
                up_cnt = up_cnt + 1
            else:
                down_neighbors.append({'idx': nei, 'area': stats[nei].area})
                down_cnt = down_cnt + 1
        else:
            if x2 > x1:
                right_neighbors.append({'idx': nei, 'area': stats[nei].area})
                right_cnt = right_cnt + 1
            else:
                left_neighbors.append({'idx': nei, 'area': stats[nei].area})
                left_cnt = left_cnt + 1

    # sort them by area and get the first 2
    up_neighbors.sort(key=lambda x: x['area'], reverse=True)
    up_neighbors = up_neighbors[:2]
    up_neighbors = [x['idx'] for x in up_neighbors]

    down_neighbors.sort(key=lambda x: x['area'], reverse=True)
    down_neighbors = down_neighbors[:2]
    down_neighbors = [x['idx'] for x in down_neighbors]

    right_neighbors.sort(key=lambda x: x['area'], reverse=True)
    right_neighbors = right_neighbors[:2]
    right_neighbors = [x['idx'] for x in right_neighbors]

    left_neighbors.sort(key=lambda x: x['area'], reverse=True)
    left_neighbors = left_neighbors[:2]
    left_neighbors = [x['idx'] for x in left_neighbors]
    all_neighbors = [up_neighbors, down_neighbors,
                     right_neighbors, left_neighbors]

    for i in range(len(all_neighbors)):
        # if there is no any neighbor
        if len(all_neighbors[i]) == 0:
            all_neighbors[i] = [superpix, superpix]
            # if there is only 1 neighbor
        if len(all_neighbors[i]) == 1:
            all_neighbors[i].append(all_neighbors[i][0])

    return all_neighbors[0] + all_neighbors[1] + all_neighbors[2] + all_neighbors[3]


def trainEoT():
    imgs_path = 'C:\\Users\\yusuf\\Desktop\\bilkent ms\\cs 579\\project\\NTU-Wrist-Image-Database-v1\\SEToriginalWristImages\\SET1'
    masks_path = 'C:\\Users\\yusuf\\Desktop\\bilkent ms\\cs 579\\project\\NTU-Wrist-Image-Database-v1\\SETsegmentedWristImages\\SET1\\mask'
    imgs = listdir(imgs_path)
    X1 = np.array([])
    X2 = np.array([])

    start_time = time.time()
    for img in imgs[:10]:
        mask_file_name = 'mask' + img[:-6] + '_binTree' + img[-6:]
        mask = read_img(masks_path + '\\' + mask_file_name)
        img_path = imgs_path + '\\' + img
        x1 = np.array(getSuperpixelFeaturesFromRegion(img_path, mask != 0))
        x2 = np.array(getSuperpixelFeaturesFromRegion(img_path, mask == 0))

        if X1.shape[0] == 0:
            X1 = x1
        else:
            X1 = np.concatenate((X1, x1))
        if X2.shape[0] == 0:
            X2 = x2
        else:
            X2 = np.concatenate((X2, x2))

        print(time.time() - start_time)

    Y1 = np.ones((X1.shape[0], ))
    Y2 = np.zeros((X2.shape[0], ))
    print('y1 shape: ' + str(Y1.shape[0]))
    print('y2 shape: ' + str(Y2.shape[0]))
    Y = np.concatenate((Y1, Y2))
    X = np.concatenate((X1, X2))

    seed, fold_count = 7, 10
    kfold = model_selection.KFold(n_splits=fold_count, random_state=seed)
    cart = DecisionTreeClassifier()
    num_trees = 10
    model = BaggingClassifier(
        base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print('-------------------------------------')
    print(results.mean())

# read image and make height 200px


def read_img(fname, img_height=200):
    img = plt.imread(fname)
    # preserve aspect ratio during resize
    wid = ceil(img_height * img.shape[1] / img.shape[0])
    return resize(img, (img_height, wid), preserve_range=True)


def getSuperpixelFeaturesFromRegion(fname, binaryCropMask):
    img_height = 200
    img = read_img(fname, img_height)

    num_superpixel = 200
    compactness = 8
    segments = slic(img, n_segments=num_superpixel, compactness=compactness)

    # plt.imshow(mark_boundaries(img, segments))
    # plt.show()

    adj_matrix = get_adjacency_matrix(segments)
    # region props ignore labels with 0 but slic labels are 0 indexed
    stats = regionprops(segments + 1)

    return extract_superpixel_features(stats, img, adj_matrix, binaryCropMask)


def test_adj_matrix(n):

    imgs_path = 'C:\\Users\\yusuf\\Desktop\\bilkent ms\\cs 579\\project\\NTU-Wrist-Image-Database-v1\\SEToriginalWristImages\\SET1'
    imgs = listdir(imgs_path)
    curr_img = imgs[randint(0, len(imgs)-1)]
    img = read_img(imgs_path + '\\' + curr_img)
    num_superpixel, compactness = 200, 8
    segments = slic(img, n_segments=num_superpixel, compactness=compactness)

    adj_matrix = get_adjacency_matrix(segments)

    for _ in range(n):
        img2 = np.array(img)
        a_segment = randint(0, len(adj_matrix)-1)
        nei = np.nonzero(adj_matrix[a_segment, :])[0]
        pic = np.full(segments.shape, False, dtype=bool)
        for n in nei:
            pic = pic | (segments == n)

        # make this superpixel red
        img2[segments == a_segment] = np.array([1, 0, 0])
        # make neigbors blue
        img2[pic] = np.array([0, 0, 1])
        plt.title(str(imgs[randint(0, len(imgs)-1)]) +
                  ' superpixel: ' + str(a_segment))
        plt.imshow(mark_boundaries(img2, segments))
        plt.show()

test_adj_matrix(5)
fname = 'img\\SEToriginalWristImages\\SET1\\0001_01_01_02_863_695_288_408_L.jpg'
# getSuperpixelFeaturesFromFile(fname)
# trainEoT()
