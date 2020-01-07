from skimage.filters import sobel_v
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale

DB_PATH = 'D:\\\\yusuf\\\\cs 579\\\\project\\\\NTU-Wrist-Image-Database-v1'


def get_img_weight(img):
    """ to show vertical wrinkles apply vertical sobel filter """
    Gx = np.array([])
    # for each image channel
    for i in range(img.shape[2]):
        if i == 0:
            Gx = abs(sobel_v(img[:, :, i]))
        else:
            Gx = np.maximum(Gx, abs(sobel_v(img[:, :, i])))

    Gx = 1 - Gx
    Gx[np.isnan(Gx)] = 0
    Gx[np.isinf(Gx)] = 0

    # img should have 200px height, reduce it to 40px
    Gx = rescale(Gx, 0.2, multichannel=False)
    return Gx


def get_up_down_boundaries(mask: np.array):
    """ return row indices of up and down lines, corresponds to maskDetermineEdges in paper code """
    _, n = mask.shape

    r = np.ones((2, 2 * n), dtype=int)
    idx_arr = np.array(range(n)).T
    r[0] = np.concatenate((idx_arr, idx_arr))

    for i in range(n):
        idxs = np.nonzero(mask[:, i])[0]
        r[1][i] = np.min(idxs)
        r[1][i + n] = np.max(idxs)
    return r


def build_graph(img, mask, connectivity, diag_dist, power_param):
    """ corresponds to 'function G = maskBuildGraph(I,mask,conn, diagParamD,powerParam)'
            img should be weighted image """
    if connectivity != 4 or connectivity != 8:
        print('connectivity must be 4 or 8, it is being set to 4')
        connectivity = 4


def test_get_img_weight():
    img = plt.imread(
        'D:\\yusuf\\cs 579\\project\\NTU-Wrist-Image-Database-v1\\SETsegmentedWristImages\\SET1\\img\\0001_01_01_01_809_866_318_423_binTree_L.jpg')
    get_img_weight(img)


def test_get_up_down_boundaries():
    mask = plt.imread(
        'D:\\yusuf\\cs 579\\project\\NTU-Wrist-Image-Database-v1\\SETsegmentedWristImages\\SET1\\mask\\mask0001_01_01_01_809_866_318_423_binTree_L.jpg')
    mask = rescale(mask, 0.2, multichannel=False,
                   preserve_range=True, anti_aliasing=False)
    bounds = get_up_down_boundaries(mask)
    mask[bounds[1], bounds[0]] = 150
    plt.imshow(mask)
    plt.show()


# test_get_img_weight()
test_get_up_down_boundaries()
