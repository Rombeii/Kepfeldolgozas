import cv2 as cv
import numpy as np

import Walsh
import Preprocessor


def display_Walsh():
    generated_hadamard = Walsh.generate_hadamard(4096)
    walsh = Walsh.convert_to_Walsh(generated_hadamard)[:512, :512]
    walsh_matrix = Walsh.get_blocks(walsh.astype(np.uint8)).reshape(8, 8, 64, 64)
    rows = walsh_matrix.shape[0]
    cols = walsh_matrix.shape[1]

    for row in range(rows):
        for col in range(cols):
            cv.imwrite("output/Walsh/{}_{}.png".format(row, col), walsh_matrix[row][col])


def test_binarization_options(img):
    converted_image = Preprocessor.convert_to_greyscale(img)
    for i in range(80, 240, 10):
        x = converted_image.copy()
        Preprocessor.binarize(x, i)
        cv.imwrite(str(i) + '.png', x)


def show_extracted_letters(letters):
    letters[0].show('elso')
    letters[1].show('masodik')
    letters[2].show('harmadik')
    letters[3].show('negyedik')
    letters[4].show('otodik')

    letters[0].write('elso_original.png')
    letters[0].resize()
    letters[0].write('elso_resized.png')
    letters[1].write('masodik.png')
    letters[2].write('harmadik.png')
    letters[3].write('negyedik.png')
    letters[4].write('otodik.png')