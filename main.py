import cv2 as cv
import Preprocessor
import Segmenter
import Walsh


def generate_etalon_vector(walsh_matrix):
    pdf = cv.imread('resources/abc.jpg')

    preprocessed_image = Preprocessor.preprocess(pdf, False)

    cv.imwrite('output/Preprocessed image_ABC.png', preprocessed_image)

    # ez akkor lesz használható, ha már tudja kezelni az i,j tetején lévő pontot
    # used_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    #                 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    #                 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    #                 'z', '.', '\\,', '!', '?', ';', '\'']

    used_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                    'skip', 'i', 'skip', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                    'w', 'x', 'y', 'z', '.', '\\,', '!', 'skip', '?', 'skip', 'skip', ';', '\'', '(', ')']

    letters = Segmenter.extract_letters(preprocessed_image)

    etalon_vector = []

    for index, letter in enumerate(letters):
        letter.resize()
        cv.imwrite('output/letters/ABC/{}_{}.png'.format(index, used_symbols[index]), letter.img)
        if used_symbols[index] != 'skip':
            etalon_vector.append((used_symbols[index], Walsh.generate_feature_vector(letter.img, walsh_matrix)))

    # print(Walsh.generate_feature_vector(letters[0].img, Walsh_matrix))
    return etalon_vector


if __name__ == '__main__':
    pdf = cv.imread('resources/fenykep.jpg')
    Walsh_matrix = Walsh.get_matrix()
    etalon_vector = generate_etalon_vector(Walsh_matrix)

    preprocessed_image = Preprocessor.preprocess(pdf)

    cv.imwrite('output/Preprocessed image_Kep.png', preprocessed_image)

    letters = Segmenter.extract_letters(preprocessed_image)

    for index, letter in enumerate(letters):
        letter.resize()
        predicted = Walsh.get_letter_prediction(Walsh.generate_feature_vector(letter.img, Walsh_matrix), etalon_vector)
        letter.write('output/letters/Kep/{}_{}.png'.format(index, predicted))

    cv.waitKey(0)