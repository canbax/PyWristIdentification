from skimage.filters import sobel_v
import numpy as np

def get_img_weight(img):
    
    Gx = np.array([])
    # for each image channel
    for i in range(img.shape[2]):
        if i == 0:
            Gx = sobel_v(img[:, :, i])
        else:
            Gx = np.maximum(Gx, sobel_v(img[:, :, i]))
    
    Gx = 1 - Gx
    Gx[np.isnan(Gx)] = 0
    Gx[np.isinf(Gx)] = 0
    
    return Gx

def test_get_img_weight():
    