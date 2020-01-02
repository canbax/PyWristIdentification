import numpy as np


# int to np array of binary integers
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
                table[i] = numPattern -1

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

    return table, numPattern


def get_grid_params(img, mask, num_ver_block, num_hor_block):
    m, n = mask.shape
    
    j_up = 0
    idx_up = 0
    idx_down = 0
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
    
    threshold2 = 
    
    
    return idx_up, idx_down, ver_step, hor_step


def test_lbp_mapping():
    print(get_lbp_mapping(8, 'riu2'))

