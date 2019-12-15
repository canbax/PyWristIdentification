from skimage.filters import sobel_v
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale

DB_PATH = 'D:\\\\yusuf\\\\cs 579\\\\project\\\\NTU-Wrist-Image-Database-v1'

# to show vertical wrinkles apply vertical sobel filter
def get_img_weight(img):

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

def get_up_down_boundaries(img):
    pass


def test_get_img_weight():
    img = plt.imread('D:\\yusuf\\cs 579\\project\\NTU-Wrist-Image-Database-v1\\SETsegmentedWristImages\\SET1\\img\\0001_01_01_01_809_866_318_423_binTree_L.jpg')
    get_img_weight(img)



# test_get_img_weight()