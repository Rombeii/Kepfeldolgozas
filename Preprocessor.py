import time
import numpy as np


def preprocess(img, do_filter=True):
    print("Preprocessing started")
    start = time.time()

    print("\tConverting to greyscale")
    converted_image = convert_to_greyscale(img)
    end = time.time()
    print("\t\tConverted to greyscale in {:.2f} seconds".format(end - start))
    start = end

    print("\tBinarizing image")
    binarize(converted_image, 155)
    end = time.time()
    print("\t\tBinarized in {:.2f} seconds".format(end - start))
    start = end

    if do_filter:
        print("\tFiltering anomalies")
        filter_anomaly(converted_image)
        end = time.time()
        print('\t\tAnomalies filtered in {:.2f} seconds'.format(end - start))
    return converted_image


def convert_to_greyscale(img):
    cols = len(img)
    rows = len(img[0])
    converted_image = np.empty([cols, rows], dtype=np.uint8)

    for i in range(cols):
        for j in range(rows):
            converted_image[i][j] = int(img[i][j][0] * 0.2126 + img[i][j][1] * 0.7152 + img[i][j][2] * 0.0722)

    return converted_image


def binarize(img, val):
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            pixel_val = img[i, j]
            img[i, j] = 255 if pixel_val > val else 0


def filter_anomaly(img):
    rows, cols = img.shape
    counter = 0

    for i in range(rows):
        for j in range(cols):
            n = neighbors(img, i, j, rows, cols)
            if sum(n == img[i, j]) == 1:
                img[i, j] = 255 if img[i, j] == 0 else 0
                counter += 1

    print('\t\t{0} values changed'.format(counter))


def neighbors(im, i, j, i_end, j_end):
    n = im[max(i - 1, 0):min(i + 2, i_end), max(j - 1, 0):min(j + 2, j_end)].flatten()
    return n
