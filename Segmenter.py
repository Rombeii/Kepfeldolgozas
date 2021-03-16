import time
import cv2 as cv


def extract_letters(img):
    print("\nSegmenting started")
    start = time.time()

    print("\tExtracting rows")
    rows = segment_horizontally(img)
    end = time.time()
    print('\t\t{} rows extracted in {:.2f} seconds'.format(len(rows), end - start))

    cv.imshow('elso_sor', rows[0])
    cv.imwrite('elso_sor.png', rows[0])

    print("\tExtracting letters")
    letters = []
    for row in rows:
        letters.extend(segment_vertically(row))
    end = time.time()
    print('\t\t{} letters extracted in {:.2f} seconds'.format(len(letters), end - start))

    print("\tCleaning letters")
    letters_clean = []
    for index, letter in enumerate(letters):
        cleaned = segment_horizontally(letter)
        letters_clean.extend(cleaned)
    end = time.time()
    print('\t\t{} letters cleaned in {:.2f} seconds'.format(len(letters_clean), end - start))

    return letters_clean


def segment_horizontally(img):
    rows, cols = img.shape
    returned = []
    # feketevel_kezdodik = all(x == 0 for x in img[0])
    utolso_feher_sor = -1

    for i in range(rows):
        row = img[i, :]
        if all(x == 255 for x in row):                                      # ha fehér a sor
            if utolso_feher_sor + 1 != i and i != 0:                        # ha már volt fekete 2 fehér között
                returned.append(img[utolso_feher_sor + 1:i, :])
            utolso_feher_sor = i

    return returned


def segment_vertically(img):
    rows, cols = img.shape
    returned = []
    utolso_feher_oszlop = -1

    for i in range(cols):
        col = img[:, i]
        if all(x == 255 for x in col):
            if utolso_feher_oszlop + 1 != i and i != 0:
                returned.append(img[:, utolso_feher_oszlop + 1:i])
            utolso_feher_oszlop = i

    return returned
