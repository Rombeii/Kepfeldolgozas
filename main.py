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
    cv.imshow('elso', letters[0])
    cv.imshow('masodik', letters[1])
    cv.imshow('harmadik', letters[2])
    cv.imshow('negyedik', letters[3])
    cv.imshow('otodik', letters[4])

    resized = cv.resize(letters[0], dsize=(64, 64), interpolation=cv.INTER_NEAREST)

    cv.imwrite('resized.png', resized)
    cv.imwrite('elso.png', letters[0])
    cv.imwrite('masodik.png', letters[1])
    cv.imwrite('harmadik.png', letters[2])
    cv.imwrite('negyedik.png', letters[3])
    cv.imwrite('otodik.png', letters[4])

    cv.waitKey(0)
