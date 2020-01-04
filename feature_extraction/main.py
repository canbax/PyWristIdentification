import numpy as np
from scipy import stats, signal
from skimage.morphology import erosion
from skimage.util import img_as_float
import cv2
from os import listdir
import matplotlib.pyplot as plt

# int to np array of binary integers
DB_PATH = 'D:\\yusuf\\cs 579\\project\\NTU-Wrist-Image-Database-v1'


def bin2dec(arr):
    v = 0
    base = 0
    for i in range(len(arr) - 1, -1, -1):
        v = v + arr[i] * (2 ** base)
        base = base + 1
    return v

# int to np array of binary integers


def dec2bin(n, width):
    s = np.binary_repr(n, width)
    return np.array([int(x) for x in s])


def test_bin2dec():
    for i in range(64):
        b = dec2bin(i, 8)
        d = bin2dec(b)
        if d != i:
            print(' NOOOO ' + str(i))
    print('If this is the only thing printed its OK!')


def get_lbp_mapping(sample_cnt, mapping_type):
    """  mapping_type can be either 'u2', 'ri' or 'riu2' """
    table = np.array(range(0, 2 ** sample_cnt))
    numPattern = 0  # number of patterns in the resulting LBP code
    idx = 0

    # uniform 2 (at max 2 0-1 or 1-0 transitions are allowed)
    if mapping_type == 'u2':
        numPattern = sample_cnt * (sample_cnt - 1) + 3
        for i in range(0, 2 ** sample_cnt):

            i_bin = dec2bin(i, sample_cnt)
            j_bin = np.roll(i_bin, -1)
            numTransition = np.sum(i_bin != j_bin)

            if numTransition <= 2:
                table[i] = idx
                idx = idx + 1
            else:
                table[i] = numPattern - 1

    # Rotation invariant
    if mapping_type == 'ri':
        tmp_map = np.zeros((2 ** sample_cnt, 1), dtype=int) - 1
        for i in range(2 ** sample_cnt):
            rm = i

            r_bin = dec2bin(i, sample_cnt)

            for j in range(sample_cnt):
                r = bin2dec(np.roll(r_bin, -1 * j))
                if r < rm:
                    rm = r
            if tmp_map[rm] < 0:
                tmp_map[rm] = numPattern
                numPattern = numPattern + 1
            table[i] = tmp_map[rm]

    # Uniform & Rotation invariant
    if mapping_type == 'riu2':
        numPattern = sample_cnt + 2
        for i in range(2 ** sample_cnt):
            i_bin = dec2bin(i, sample_cnt)
            j_bin = np.roll(i_bin, -1)
            numTransition = sum(i_bin != j_bin)

            if numTransition <= 2:
                table[i] = sum(i_bin)
            else:
                table[i] = sample_cnt + 1

    return {'table': table, 'numPattern': numPattern}


def get_grid_params(img, mask, num_ver_block, num_hor_block):
    """ returns idx_up, idx_down, ver_step, hor_step """
    m, n = mask.shape

    j_up = 0
    idx_up = 0
    idx_down = 0
    idx_left = 0
    idx_right = n
    ver_step = 0
    hor_step = 0
    temp = 0
    threshold1 = np.mean(mask)
    # find up index
    while temp < threshold1:
        j_up = j_up + 1
        temp = np.mean(mask[j_up, :])
        idx_up = j_up

    temp = 0
    j_down = m
    # find down index
    while temp < threshold1:
        temp = np.mean(mask[j_down, :])
        j_down = j_down - 1
        idx_down = j_down

    threshold2 = np.mean(stats.mode(mask[idx_up:idx_down, :]))
    i_left = 0
    temp = 0
    while temp < threshold2:
        i_left = i_left + 1
        temp = np.mean(mask[idx_up:idx_down, i_left])
        idx_left = i_left

    temp = 0
    i_right = n
    while temp < threshold2:
        i_right = i_right - 1
        temp = np.mean(mask[idx_up:idx_down, i_right])
        idx_right = i_right

    j_up = 0
    j_down = m
    threshold1 = np.mean(stats.mode(mask[:, idx_left:idx_right]))
    temp = 0
    while temp < threshold1:
        j_up = j_up + 1
        temp = np.mean(mask[j_up, idx_left:idx_right])
        idx_up = j_up

    temp = 0
    while temp < threshold1:
        temp = np.mean(mask[j_down, idx_left:idx_right])
        j_down = j_down - 1
        idx_down = j_down

    hor_step = np.floor((idx_down - idx_up) / num_hor_block)
    ver_step = np.floor(n / num_ver_block)

    idx_up = idx_up - \
        np.floor((idx_up + num_hor_block * hor_step - idx_down) / 2)
    if idx_up < 1:
        idx_up = 1

    return idx_up, idx_down, ver_step, hor_step


def maskimage(im, mask, col=0):
    _, _, chan = im.shape
    col = np.ones((chan, 1))

    maskedim = im
    for n in range(chan):
        tmp = maskedim[:, :, n]
        tmp[mask] = col[n]
        maskedim[:, :, n] = tmp

    return maskedim


def extract2(img: np.array, mask: np.array, settings: dict, gray: np.array):
    mappings = [settings['map1'], settings['map2']]
    radiuses = [settings['radius1'], settings['radius2']]
    neighb = settings['neighb']

    mask4lbp = erosion(mask, np.ones((5, 5)))
    features = []
    for c in range(img.shape[2]):
        for i in range(2):
            r = radiuses[i]
            m = mappings[i]
            F_C_ = lbp(img[:, :, c], r, neighb, m)
            F_C_ = F_C_[r + 1:-r, r + 1:-r]
            F_C_ = F_C_ + 1
            F_C_ = maskimage(F_C_, np.logical_not(mask4lbp))

            bin_nums = [10, 10, 59, 59, 59, 59, 59]
            for j in range(len(settings['grids'])):
                f = extract_lbp(F_C_, settings['grids'][j], bin_nums[j])
                features.append(f.reshape((1, f.size)))

    gray2 = (0.2989 * img[:, :, 0]) + (0.5870 *
                                       img[:, :, 1]) + (0.1140 * img[:, :, 2])

    s = [.2, .5, .7, .9]
    rp = 1
    ImgGabor = np.pad(gray2, rp, mode='constant')
    maskGabor = np.pad(mask, rp, mode='constant')
    Orient_Map = vein_gabor_enhancement(ImgGabor, s)

    Orient_Map = Orient_Map[rp+1: -rp, rp+1: -rp]
    maskGabor = erosion(maskGabor, get_disk8_filter())
    maskGabor = maskGabor[rp + 1:-rp, rp + 1: -rp]
    Orient_Map = Orient_Map + 1
    Orient_Map = maskimage(Orient_Map, np.logical_not(maskGabor))

    for j in range(len(settings['grids'])):
        f = extract_gabor(Orient_Map, settings['grids'][j])
        features.append(f.reshape((1, f.size)))

    return np.array(features)


def get_disk8_filter():
    arr_size = 15
    r = np.ones((arr_size, arr_size))
    for i in range(4):
        for j in range(4):
            if i + j < 4:
                r[i, j] = 0
                f = arr_size - 1
                r[f - i, j] = 0
                r[i, f - j] = 0
                r[f - i, f - j] = 0

    return r


def lbp(img, radius, neighb, mapping):
    img = np.pad(img, radius * 2, mode='constant')
    spoints = np.zeros((neighb, 2))

    # Angle step.
    a = 2 * np.pi / neighb

    for i in range(neighb):
        spoints[i, 0] = -radius * np.sin(i * a)
        spoints[i, 1] = radius * np.cos(i * a)

    miny = min(spoints[:, 0])
    maxy = max(spoints[:, 0])
    minx = min(spoints[:, 1])
    maxx = max(spoints[:, 1])

    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    bsizey = np.ceil(max(maxy, 0)) - np.floor(min(miny, 0)) + 1
    bsizex = np.ceil(max(maxx, 0)) - np.floor(min(minx, 0)) + 1

    # Coordinates of origin (0,0) in the block
    origy = 1 - np.floor(min(miny, 0))
    origx = 1 - np.floor(min(minx, 0))

    ysize, xsize = img.shape

    # Calculate dx and dy
    dx = xsize - bsizex
    dy = ysize - bsizey

    # Fill the center pixel matrix C.
    C = img[origy:origy + dy, origx:origx + dx]
    d_C = C.astype(np.double)

    # Initialize the result matrix with zeros.
    result = np.zeros((dy + 1, dx + 1))
    d_image = img.astype(np.double)
    for i in range(neighb):
        y = spoints[i, 1] + origy
        x = spoints[i, 2] + origx

        # Calculate floors, ceils and rounds for the x and y.
        fy = np.floor(y)
        cy = np.ceil(y)
        ry = np.round(y)
        fx = np.floor(x)
        cx = np.ceil(x)
        rx = np.round(x)

        # Check if interpolation is needed
        E = 1e-6
        if abs(x - rx) < E and abs(y - ry) < E:
            # Interpolation is not needed, use original datatypes
            N = img[ry:ry + dy, rx:rx + dx]
            D = N >= C
        else:
            # Interpolation needed, use double type images
            ty = y - fy
            tx = x - fx

            # Calculate the interpolation weights.
            w1 = roundn((1 - tx) * (1 - ty), -6)
            w2 = roundn(tx * (1 - ty), -6)
            w3 = roundn((1 - tx) * ty, -6)
            w4 = roundn(1 - w1 - w2 - w3, -6)

            # Compute interpolated pixel values
            N = w1 * d_image[fy:fy + dy, fx:fx + dx] + w2 * d_image[fy:fy + dy, cx:cx + dx] + \
                w3 * d_image[cy:cy + dy, fx: fx + dx] + \
                w4 * d_image[cy: cy + dy, cx: cx + dx]
            N = roundn(N, -4)
            D = N >= d_C
        # Update the result matrix.
        v = 2 ** (i-1)
        result = result + v * D

    # apply mapping
    # bins = mapping['numPattern']
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = mapping['table'][result[i, j]]

    return result.astype(np.uint32)


def roundn(x, n):
    x = 0
    if n < 0:
        p = 10 ^ -n
        x = round(p * x) / p
    elif n > 0:
        p = 10 ^ n
        x = p * round(x / p)
    else:
        x = round(x)
    return x


def extract_lbp(F1: np.array, grid: dict, bin_num):
    num_hor_block = grid['num_hor_block']
    num_ver_block = grid['num_ver_block']
    hor_step = grid['hor_step']
    ver_step = grid['ver_step']
    idx_up = grid['idx_up']
    histMatrix = np.zeros(num_hor_block * num_ver_block, bin_num)

    indM = 0
    y = 1
    x = idx_up
    for _ in range(num_hor_block):
        x = idx_up
        for _ in range(num_ver_block):
            indM = indM + 1
            box = F1[x:x + hor_step, y:y + ver_step]
            histMatrix[indM, :], _ = np.histogram(
                box[box != 0], range(1, bin_num + 1))
            x = x + hor_step
        y = y + ver_step

    return histMatrix


def vein_gabor_enhancement(img, scale):
    I = img_as_float(img)
    No_of_Orientation = 16  # in wrist paper 16 oreientations
    No_of_Scale = len(scale)
    a = scale
    Gr, Gi = Gabor(0, 1.5, 1, 200, a[No_of_Scale-1])  # 1.5
    p = Energy_check(Gr)
    p_1 = p+1

    Gabor_Matrix_Gr = np.zeros(
        (2*p + 1, 2*p + 1, No_of_Orientation, No_of_Scale))

    ER = [0] * No_of_Scale

    for s in range(No_of_Scale):
        ang_count = 0
        for ang in range(0, 179, 180 / No_of_Orientation):
            ang_count = ang_count + 1
            Gr, Gi = Gabor(ang, 1.5, 1, p, a[s])  # 1.5
            Gabor_Matrix_Gr[:, :, ang_count, s] = Gr
        R = Energy_check(np.squeeze(Gabor_Matrix_Gr[:, :, 1, s]))
        ER[s] = R

    Energy_Map = np.zeros(I.shape)
    Scale_Map = np.ones(I.shape)
    Orient_Map = np.ones(I.shape)

    E = np.zeros((I.shape[0], I.shape[1], No_of_Orientation, No_of_Scale))

    Gabor_fft_Matrix_Gr = np.zeros(
        (I.shape[0], I.shape[1], No_of_Orientation, No_of_Scale))

    I_fft = np.fft.fft2(I)
    for s in range(No_of_Scale):
        for ang in range(No_of_Orientation):
            pad = np.floor([(I.shape[0] - (2*p+1))/2, (I.shape[1]-(2*p+1))/2])
            Gabor_fft_Matrix_Gr[:, :, ang, s] = np.fft.fft2(
                np.pad(np.squeeze(Gabor_Matrix_Gr[:, :, ang, s]), pad, mode='constant'), I.shape[0], I.shape[1])

    I2 = np.square(I)
    Image_Power = np.zeros((I.shape[0], I.shape[1], No_of_Scale))
    for s in range(No_of_Scale):
        Image_Power[:, :, s] = np.sqrt(
            signal.convolve2d(np.ones(ER[s]), I2, 'same')) + 0.1
        for ang in range(No_of_Orientation):
            E[:, :, ang, s] = np.fft.fftshift(np.fft.ifft2(-np.squeeze(
                Gabor_fft_Matrix_Gr[:, :, ang, s]) * I_fft)) / np.squeeze(Image_Power[:, :, s])

    EngAng = np.zeros(I.shape[0], I.shape[1], No_of_Scale)
    AngIdx = np.zeros(I.shape[0], I.shape[1], No_of_Scale)
    for s in range(No_of_Scale):
        EngAng[:, :, s] = np.amax(E[:, :, :, s], axis=2)
        AngIdx[:, :, s] = np.argmax(E[:, :, :, s], axis=2)
    Energy_Map = np.amax(EngAng, axis=2)
    Scale_Map = np.argmax(EngAng, axis=2)
    for x in range(Energy_Map.shape[0]):
        for y in range(Energy_Map.shape[1]):
            Orient_Map[x, y] = AngIdx[x, y, Scale_Map[x, y]]

    return Orient_Map


def Energy_check(G):

    Total_energy = np.sum(G*G) ** 0.5
    m = G.shape[0]
    S = (m-1) / 2
    cx = (m-1) / 2
    cy = (m-1) / 2
    R = S
    for x in range(S):
        Energy = np.sum(np.square(G[cx - x:cx + x, cy - x:cy + x])) ** 0.5
        Energy = Energy / Total_energy * 100
        if Energy > 99.9:
            R = x
            break
    return R


def Gabor(ang, d, w, N, a):
    """ Gabor(ang, 1.5, 1, p, a(s))
    ang: rotation angle, d: bandwidth, w: wavelength,N: Filter size, a: scale """

    k = (2 * np.log(2))**0.5 * ((2**d + 1) / (2**d - 1))
    Gr = np.zeros((2*N + 1, 2*N + 1))
    Gi = np.zeros((2*N + 1, 2*N + 1))
    ang_d = ang * np.pi / 180
    COS = np.cos(ang_d)
    SIN = np.sin(ang_d)
    const = w / ((2 * np.pi) ** 0.5 * k)

    for x in range(-N, N + 1):
        for y in range(-N, N + 1):
            x_1 = x * COS + y * SIN
            y_1 = -x * SIN + y * COS
            x_1 = x_1 / a
            y_1 = y_1 / a
            temp = 1 / const * np.exp(-w * w / (8*k*k) * (4 * x_1**2 + y_1**2))
            Gr[x+N+1, y+N+1] = a ** -1 * temp * \
                (np.cos(w*x_1) - np.exp(-k * k/2))
            Gi[x+N+1, y+N+1] = a ** -1 * temp * np.sin(w * x_1)

    return Gr, Gi


def extract_gabor(orient_map: np.array, grid: dict):
    num_ver_block = grid['num_ver_block']
    num_hor_block = grid['num_hor_block']
    hor_step = grid['hor_step']
    ver_step = grid['ver_step']
    ind_up = grid['idx_up']
    histMatrix = np.zeros((num_ver_block * num_hor_block, 16))

    indM = 0

    y = 1
    for _ in range(num_hor_block):
        x = ind_up
        for _ in range(num_ver_block):
            indM = indM + 1
            box = orient_map[x:x + hor_step, y:y + ver_step]
            histMatrix[indM, :] = np.histogram(box(box != 0), range(1, 17))
            x = x + hor_step
        y = y + ver_step

    return histMatrix


def build_feature_vectors():
    imgs_path = DB_PATH + '\\SETsegmentedAlignedWristImages\\SET1\\img'
    masks_path = DB_PATH + '\\SETsegmentedAlignedWristImages\\SET1\\mask'
    imgs = listdir(imgs_path)

    num_training_img = 10
    for img in imgs[:num_training_img]:
        mask_file_name = 'mask' + img[:-6] + '_binTree' + img[-6:]
        mask = plt.imread(masks_path + '\\' + mask_file_name)
        I = plt.imread(imgs_path + '\\' + img)
        

def get_sift_features(gray: np.array):
    # sift = cv2.xfeatures2d.SIFT_create()
    # _, des = sift.detectAndCompute(gray, None)
    a = 1


def test_lbp_mapping():
    print(get_lbp_mapping(8, 'riu2'))
