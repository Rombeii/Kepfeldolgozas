import cv2 as cv
import Preprocessor
import Segmenter
import Walsh
import numpy as np

if __name__ == '__main__':
    pdf = cv.imread('resources/abc.jpg')

    preprocessed_image = Preprocessor.preprocess(pdf, False)

    # cv.imshow('Preprocessed image', preprocessed_image)
    cv.imwrite('output/Preprocessed image.png', preprocessed_image)

    # ez akkor lesz használható, ha már tudja kezelni az i,j tetején lévő pontot
    # used_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    #                 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    #                 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    #                 'z', '.', '\\,', '!', '?', ';', '\'']

    used_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                    'skip', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                    'w', 'x', 'y', 'z', '.', '\\,', '!', 'skip', '?', 'skip', 'skip', ';', '\'', '(', ')']
    letters = Segmenter.extract_letters(preprocessed_image)

    etalon_vector = []

    Walsh_matrix = Walsh.get_matrix()
    for index, letter in enumerate(letters):
        letter.resize()
        cv.imwrite('output/letters/{}.png'.format(index), letter.img)
        if used_symbols[index] != 'skip':
            etalon_vector.append((used_symbols[index], Walsh.generate_feature_vector(letter.img, Walsh_matrix)))

    # print(Walsh.generate_feature_vector(letters[0].img, Walsh_matrix))

    cv.waitKey(0)
