import cv2 as cv
import Preprocessor
import Segmenter

if __name__ == '__main__':
    pdf = cv.imread('fenykep.jpg')

    cv.imshow('Original', pdf)

    # Preprocessor.test_binarization_options(pdf)

    preprocessed_image = Preprocessor.preprocess(pdf, True)

    cv.imshow('Preprocessed image', preprocessed_image)
    cv.imwrite('Preprocessed image.png', preprocessed_image)

    letters = Segmenter.extract_letters(preprocessed_image)
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

    cv.waitKey(0)
