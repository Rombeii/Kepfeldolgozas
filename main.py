import cv2 as cv
import Preprocessor
import Segmenter
import Walsh
import numpy as np

if __name__ == '__main__':
    pdf = cv.imread('fenykep.jpg')

    cv.imshow('1', Walsh.get_matrix()[0][0])
    cv.imshow('2', Walsh.get_matrix()[0][1])
    cv.imshow('3', Walsh.get_matrix()[0][2])
    cv.imshow('5', Walsh.get_matrix()[1][0])
    cv.imshow('6', Walsh.get_matrix()[2][0])
    cv.imshow('4', Walsh.generate_hadamard().astype(np.uint8)[:512, :512])

    # Preprocessor.test_binarization_options(pdf)

    # preprocessed_image = Preprocessor.preprocess(pdf, True)
    #
    # cv.imshow('Preprocessed image', preprocessed_image)
    # cv.imwrite('Preprocessed image.png', preprocessed_image)
    #
    # letters = Segmenter.extract_letters(preprocessed_image)
    # letters[0].show('elso')
    # letters[1].show('masodik')
    # letters[2].show('harmadik')
    # letters[3].show('negyedik')
    # letters[4].show('otodik')
    #
    # letters[0].write('elso_original.png')
    # letters[0].resize()
    # letters[0].write('elso_resized.png')
    # letters[1].write('masodik.png')
    # letters[2].write('harmadik.png')
    # letters[3].write('negyedik.png')
    # letters[4].write('otodik.png')

    cv.waitKey(0)
