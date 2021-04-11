from scipy.linalg import hadamard
import numpy as np


def get_matrix():
    generated_hadamard = generate_hadamard()
    walsh = convert_to_Walsh(generated_hadamard)[:512, :512]

    return get_blocks(walsh.astype(np)).reshape(8, 8, 64, 64)
    #return get_blocks(walsh.astype(np.uint8)).reshape(8, 8, 64, 64) ha meg is szeretnénk jeleníteni


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


def generate_hadamard():
    return hadamard(4096)


def generate_feature_vector(picture, walsh_matrices):

    rows = walsh_matrices.shape[0]
    cols = walsh_matrices.shape[1]
    picture[picture == 255] = 1
    feature_vector = []
    for row in range(rows):
        for col in range(cols):
            feature_vector.append(np.multiply(picture, walsh_matrices[row][col]).sum())

    return feature_vector
