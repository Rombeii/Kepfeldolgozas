from scipy.linalg import hadamard
import numpy as np


def get_matrix():
    generated_hadamard = generate_hadamard()
    walsh = convert_to_Walsh(generated_hadamard)[:512, :512]

    return get_blocks(walsh.astype(np.uint8)).reshape(8, 8, 64, 64)


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