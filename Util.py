import cv2 as cv
import numpy as np

import Walsh
import Preprocessor


def display_Walsh():
    walsh_matrix = Walsh.get_matrix()
    cv.imshow('1', walsh_matrix[0][0])
    cv.imshow('2', walsh_matrix[0][1])
    cv.imshow('3', walsh_matrix[0][2])
    cv.imshow('5', walsh_matrix[1][0])
    cv.imshow('6', walsh_matrix[2][0])
    cv.imshow('4', Walsh.generate_hadamard().astype(np.uint8)[:512, :512])


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