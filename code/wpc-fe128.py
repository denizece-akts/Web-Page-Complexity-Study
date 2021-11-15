import numpy as np
import cv2
from glob import glob
from skimage.feature import hog

imgs_train = glob("./trainval/*.png")
imgs_test = glob("./test/*.png")

col_num = int(1024 / 128)       # for 1024x1024 px image, 128x128 patch size, there are 8 columns and 64 grids in total

# find the neighbors of given cell
X = col_num - 1     # column - 1
Y = col_num - 1

neighbors = lambda x, y: [(x2, y2) for x2 in range(x - 1, x + 2) for y2 in range(y - 1, y + 2)
                          if (-1 < x <= X and -1 < y <= Y and (x != x2 or y != y2)
                              and (0 <= x2 <= X) and (0 <= y2 <= Y))]


def grid_partitioning(image, patch_size):

    patches = []
    w, h, _ = image.shape

    for x in range(0, h, patch_size):
        for y in range(0, w, patch_size):
            grid = image[x: x + patch_size, y:y + patch_size]
            patches.append(grid)

    return patches


# CLD

def representative_color(grid, patch_size): 		# calculates the representative color of given patch(grid)

    blocks = np.zeros((8, 8, 3))
    step = int(patch_size / 8)

    for r in range(8):						# divided into 64 blocks
        for c in range(8):
            block = grid[r: r + step, c:c + step]
            avg_color = np.mean(block, axis=(0, 1))
            avg_color = np.uint8(avg_color)
            blocks[r, c, :] = avg_color

    return blocks


def compute_dct(blocks):

    im = cv2.cvtColor(np.array(blocks, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB)			# changes the color space
    y, cr, cb = cv2.split(im)
    dct_y = cv2.dct(np.float32(y))
    dct_cb = cv2.dct(np.float32(cb))
    dct_cr = cv2.dct(np.float32(cr))

    return dct_y, dct_cb, dct_cr


def zigzag(dct_y, dct_cb, dct_cr):

    dct_y_zigzag = []
    dct_cb_zigzag = []
    dct_cr_zigzag = []

    flip = True

    flipped_dct_y = np.fliplr(dct_y)
    flipped_dct_cb = np.fliplr(dct_cb)
    flipped_dct_cr = np.fliplr(dct_cr)

    for i in range(15):  # 15 = 8 + 8 -1

        k_diag = 7 - i  # 8 - 1
        diag_y = np.diag(flipped_dct_y, k=k_diag)
        diag_cb = np.diag(flipped_dct_cb, k=k_diag)
        diag_cr = np.diag(flipped_dct_cr, k=k_diag)

        if flip:
            diag_y = diag_y[::-1]
            diag_cb = diag_cb[::-1]
            diag_cr = diag_cr[::-1]

        dct_y_zigzag.append(diag_y)
        dct_cb_zigzag.append(diag_cb)
        dct_cr_zigzag.append(diag_cr)

        flip = not flip

    return np.concatenate([np.concatenate(dct_y_zigzag), np.concatenate(dct_cb_zigzag), np.concatenate(dct_cr_zigzag)])


# HOG

def hog_descriptor(grid, cell_size):

    fd, hog_image = hog(grid, orientations=9,
                        pixels_per_cell=(cell_size, cell_size),
                        cells_per_block=(1, 1),
                        visualize=True,
                        multichannel=True)

    return fd, hog_image


def histogram_intersection(x, y):
    return np.sum(np.minimum(x, y))


def hik(array, cols):

    hiks = []

    for r in range(cols):
        for c in range(cols):
            nbrs = neighbors(r, c)			# returns indices of neighbors
            hik = 0
            for idx in nbrs:
                x, y = idx
                hik += histogram_intersection(array[r][c], array[x][y])
            hiks.append(hik / len(nbrs))
    return hiks


def extract_clds(imgs):
    
    cld_hiks = []

    for img in imgs:

        img = cv2.imread(img)
        grids = grid_partitioning(img, 128)

        img_cld = []
        for grid in grids:               # grid == patch
            blocks = representative_color(grid, 128)
            patch = zigzag(*compute_dct(blocks))
            img_cld.append(patch)

        img_cld = np.array([img_cld[i:i+col_num] for i in range(0, len(img_cld), col_num)])  # makes 2d array to find neighbors
        cld_hiks.append(hik(img_cld, col_num))

    return cld_hiks


def extract_hogs(imgs):
    
    hog_hiks = []

    for img in imgs:

        img = cv2.imread(img)
        grids = grid_partitioning(img, 128)

        img_hog = []
        for grid in grids:

            fd, hog_image = hog_descriptor(grid, 32)

            img_hog.append(fd)

        img_hog = np.concatenate(img_hog)
        img_hog = np.array([img_hog[i:i+col_num] for i in range(0, len(img_hog), col_num)])  # makes 2d array to find neighbors
        hog_hiks.append(hik(img_hog, col_num))
    
    return hog_hiks


cld_features_train = extract_clds(imgs_train)
cld_features_test = extract_clds(imgs_test)

hog_features_train = extract_hogs(imgs_train)
hog_features_test = extract_hogs(imgs_test)

stacked_tr = np.concatenate((hog_features_train, cld_features_train), axis=1)			# train data
stacked_test = np.concatenate((hog_features_test, cld_features_test), axis=1)			# test data

np.savetxt('train_data256.csv', stacked_tr, delimiter=',')
np.savetxt('test_data256.csv', stacked_test, delimiter=',')
