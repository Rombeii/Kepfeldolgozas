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


if __name__ == '__main__':
    pdf = cv.imread('resources/Calibri_test.jpg')
    # Util.test_binarization_options(pdf)
    Walsh_matrix = Walsh.get_matrix()
    etalon_vector = generate_etalon_vector(Walsh_matrix)

    preprocessed_image = Preprocessor.preprocess(pdf)

    cv.imwrite('output/Preprocessed image_Kep.png', preprocessed_image)

    letters = Segmenter.extract_letters(preprocessed_image)

    predicted_letters = []

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

        predicted_letters.append((predicted, letter))

    recovered_text = []
    previous_rownum = 0
    for index, prediction in enumerate(predicted_letters):
        if prediction[1].rownum > previous_rownum and prediction[0] != '':
            recovered_text.append("\n\n")

        if index != 0 and letters[index].x_coord - (letters[index - 1].x_coord + letters[index - 1].width) > 20:
            recovered_text.append(" ")

        recovered_text.append(prediction[0])
        previous_rownum = prediction[1].rownum

    skipNext = False
    clean_text = []
    for index, text in enumerate(recovered_text):
        if skipNext:
            skipNext = False
            continue
        if index != len(recovered_text) - 1 and text == '.' and recovered_text[index + 1] in ['i', '!', 'l', 'I']:
            clean_text.append('i')
            skipNext = True
        elif index != len(recovered_text) - 1 and text == '\'' and recovered_text[index + 1] == '\'':
            clean_text.append('"')
            skipNext = True
        elif index != len(recovered_text) - 1 and text == '?' and recovered_text[index + 1] == '.':
            clean_text.append('"')
            skipNext = True
        elif index != len(recovered_text) - 1 and text == 'I' and recovered_text[index + 1] == '.':
            clean_text.append('!')
            skipNext = True
        else:
            clean_text.append(text)

    with open("output/kimenet.txt", "w") as filehandle:
        for listitem in clean_text:
            filehandle.write(listitem)

    cv.waitKey(0)
