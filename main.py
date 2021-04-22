import cv2 as cv
import Preprocessor
import Segmenter
import Util
import Walsh


def generate_etalon_vector(walsh_matrix):
    pdf = cv.imread('resources/ABC_Calibri.jpg')

    preprocessed_image = Preprocessor.preprocess(pdf, False)

    cv.imwrite('output/Preprocessed image_ABC.png', preprocessed_image)

    used_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                    'skip', 'i', 'skip', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                    'w', 'x', 'y', 'z', '.', ',', '!', 'skip', '?', 'skip', 'skip', ';', '\'', '(', ')']

    letters = Segmenter.extract_letters(preprocessed_image)

    etalon_vector = []

    for index, letter in enumerate(letters):
        letter.resize()
        cv.imwrite('output/letters/ABC/{}_{}.png'.format(index, used_symbols[index]), letter.img)
        if used_symbols[index] != 'skip':
            etalon_vector.append((used_symbols[index], Walsh.generate_feature_vector(letter.img, walsh_matrix)))

    # print(Walsh.generate_feature_vector(letters[0].img, Walsh_matrix))
    return etalon_vector


def get_predicted_letters(letters):
    returned_letters = []

    for index, letter in enumerate(letters):
        letter.resize()
        feature_vector = Walsh.generate_feature_vector(letter.img, Walsh_matrix)
        predicted = Walsh.get_letter_prediction(feature_vector, etalon_vector)

        letter.write('output/letters/Kep/{}_{}_{}.png'.format(index, letter.height, predicted))

        if letter.height < 5:
            predicted = ''

        if predicted in ['i', '!', 'l', 'I', '.', '\'']:
            if letter.height < 20:
                predicted = '.'
            elif letter.height < 35:
                predicted = '\''
            elif letter.height < 55:
                predicted = 'i'
            elif letter.height < 72:
                predicted = 'I'
            else:
                predicted = 'l'

        if predicted.lower() in ['c', 'o', 's', 'v', 'w', 'x', 'z']:
            if letter.height < 60:
                predicted = predicted.lower()
            else:
                predicted = predicted.upper()

        returned_letters.append((predicted, letter))
    return returned_letters


def add_whitespaces(predicted_letters, letters):
    returned_text = []
    previous_rownum = 0
    for index, prediction in enumerate(predicted_letters):
        if prediction[1].rownum > previous_rownum and prediction[0] != '':
            returned_text.append("\n\n")

        if index != 0 and letters[index].x_coord - (letters[index - 1].x_coord + letters[index - 1].width) > 20:
            returned_text.append(" ")

        returned_text.append(prediction[0])
        previous_rownum = prediction[1].rownum

    return returned_text


def correct_letters(letters):
    skip_next = False
    returned_text = []
    for index, text in enumerate(letters):
        if skip_next:
            skip_next = False
            continue
        if index != len(letters) - 1 and text == '.' and letters[index + 1] in ['i', '!', 'l', 'I']:
            returned_text.append('i')
            skip_next = True
        elif index != len(letters) - 1 and text == '\'' and letters[index + 1] == '\'':
            returned_text.append('"')
            skip_next = True
        elif index != len(letters) - 1 and text == '?' and letters[index + 1] == '.':
            returned_text.append('"')
            skip_next = True
        elif index != len(letters) - 1 and text == 'I' and letters[index + 1] == '.':
            returned_text.append('!')
            skip_next = True
        else:
            returned_text.append(text)
    return returned_text


if __name__ == '__main__':
    pdf = cv.imread('resources/Calibri_test.jpg')
    # Util.test_binarization_options(pdf)
    Walsh_matrix = Walsh.get_matrix()
    etalon_vector = generate_etalon_vector(Walsh_matrix)

    preprocessed_image = Preprocessor.preprocess(pdf)

    cv.imwrite('output/Preprocessed image_Kep.png', preprocessed_image)

    letters = Segmenter.extract_letters(preprocessed_image)

    predicted_letters = get_predicted_letters(letters)
    letters_with_whitespaces = add_whitespaces(predicted_letters, letters)
    clean_text = correct_letters(letters_with_whitespaces)


    with open("output/kimenet.txt", "w") as filehandle:
        for listitem in clean_text:
            filehandle.write(listitem)

    cv.waitKey(0)
