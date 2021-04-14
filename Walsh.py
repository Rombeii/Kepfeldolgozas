from scipy.linalg import hadamard
import numpy as np
import sys
import math
import cv2 as cv


def get_matrix():
    generated_hadamard = generate_hadamard(12)
    walsh = convert_to_Walsh(generated_hadamard)[:512, :512]

    return get_blocks(walsh.astype(np)).reshape(8, 8, 64, 64)
    # return get_blocks(walsh.astype(np.uint8)).reshape(8, 8, 64, 64) ha meg is szeretnénk jeleníteni


def convert_to_Walsh(generated_hadamard):
    rows, cols = generated_hadamard.shape

    walsh = []
    for i in range(rows):
        positive = generated_hadamard[i] > 0
        count = len(np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0])  # megszámolja, hány sign váltás volt
        walsh.append((count, generated_hadamard[i]))
    walsh.sort()
    return np.vstack([x[1] for x in walsh]) * -1


def get_blocks(matrix):
    rows, cols = matrix.shape
    return (matrix.reshape(rows // 64, 64, -1, 64)
            .swapaxes(1, 2)
            .reshape(-1, 64, 64))


def generate_hadamard(n):
    alap = np.array([[1]], dtype=int)

    for i in range(0, n):
        alap = np.vstack((np.hstack((alap, alap)), np.hstack((alap, -alap))))

    return alap


def generate_feature_vector(picture, walsh_matrices):
    temp = np.copy(picture)
    rows = walsh_matrices.shape[0]
    cols = walsh_matrices.shape[1]
    temp[temp == 255] = 1
    feature_vector = []
    for row in range(rows):
        for col in range(cols):
            feature_vector.append(np.multiply(temp, walsh_matrices[row][col]).sum())

    return feature_vector


def get_letter_prediction(picture, etalon_vector):
    prediction = None
    min_diff = sys.maxsize
    for etalon in etalon_vector:
        diff = get_diff(picture, etalon)
        if diff < min_diff:
            prediction = etalon[0]
            min_diff = diff

    return prediction


def get_diff(picture, etalon):
    return np.sum(np.abs(np.subtract(picture, etalon[1])))
