from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np


def get_adjacency_matrix(segments: np.array):
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


def extract_features():
    pass


fname = 'img\\SEToriginalWristImages\\SET1\\0001_01_01_02_863_695_288_408_L.jpg'
num_superpixel = 200
img_height = 200
compactness = 8

img = plt.imread(fname)
img = resize(img, (img_height, img.shape[1]))
segments = slic(img, n_segments=num_superpixel, compactness=compactness)
adj_matrix = get_adjacency_matrix(segments)

print(adj_matrix)
