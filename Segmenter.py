import time
import cv2 as cv


class SegmentedImage:
    def __init__(self, x_coord, y_coord, width, height, img, rownum=0) -> None:
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.width = width
        self.height = height
        self.img = img
        self.rownum = rownum

    def show(self, name):
        cv.imshow(name, self.img)

    def write(self, name):
        cv.imwrite(name, self.img)

    def resize(self, height=64, width=64):
        self.img = cv.resize(self.img, dsize=(height, width), interpolation=cv.INTER_NEAREST)

    def set_rownum(self, rownum):
        self.rownum = rownum

def extract_letters(img):
    print("\nSegmenting started")
    start = time.time()

    print("\tExtracting rows")
    height, width = img.shape
    rows = segment_horizontally(SegmentedImage(0, 0, width, height, img))
    end = time.time()
    print('\t\t{} rows extracted in {:.2f} seconds'.format(len(rows), end - start))

    cv.imwrite('output/elso_sor.png', rows[0].img)

    for index, row in enumerate(rows):
        row.set_rownum(index)

    print("\tExtracting letters")
    letters = []
    for row in rows:
        letters.extend(segment_vertically(row))
    end = time.time()
    print('\t\t{} letters extracted in {:.2f} seconds'.format(len(letters), end - start))

    print("\tCleaning letters")
    temp = []
    letters_clean = []
    for index, letter in enumerate(letters):
        cleaned = segment_horizontally(letter)
        temp.extend(cleaned)
    for index, letter in enumerate(temp):
        cleaned = segment_vertically(letter)
        letters_clean.extend(cleaned)
    end = time.time()
    print('\t\t{} letters cleaned in {:.2f} seconds'.format(len(letters_clean), end - start))

    return letters_clean


def segment_horizontally(segmented_img):
    rows, cols = segmented_img.img.shape
    returned = []
    # feketevel_kezdodik = all(x == 0 for x in img[0])
    utolso_feher_sor = -1

    for i in range(rows):
        row = segmented_img.img[i, :]
        if all(x == 255 for x in row):                                      # ha fehér a sor
            if utolso_feher_sor + 1 != i and i != 0:                        # ha már volt fekete 2 fehér között
                returned.append(SegmentedImage(segmented_img.x_coord, segmented_img.y_coord + utolso_feher_sor + 1,
                                               segmented_img.width, i - (utolso_feher_sor + 1),
                                               segmented_img.img[utolso_feher_sor + 1:i, :], segmented_img.rownum))
            utolso_feher_sor = i

    if utolso_feher_sor != rows - 1:                                        # ha az utolsó sor fekete
        returned.append(SegmentedImage(segmented_img.x_coord, segmented_img.y_coord + utolso_feher_sor + 1,
                                       segmented_img.width, rows - (utolso_feher_sor + 1),
                                       segmented_img.img[utolso_feher_sor + 1:rows, :], segmented_img.rownum))

    if not returned:                                                        # ha nem történt vágás
        returned.append(segmented_img)

    return returned


def segment_vertically(segmented_img):
    rows, cols = segmented_img.img.shape
    returned = []
    utolso_feher_oszlop = -1

    for i in range(cols):
        col = segmented_img.img[:, i]
        if all(x == 255 for x in col):
            if utolso_feher_oszlop + 1 != i and i != 0:
                returned.append(SegmentedImage(segmented_img.x_coord + utolso_feher_oszlop + 1, segmented_img.y_coord,
                                               i - (utolso_feher_oszlop + 1), segmented_img.height,
                                               segmented_img.img[:, utolso_feher_oszlop + 1:i], segmented_img.rownum))
            utolso_feher_oszlop = i

    if utolso_feher_oszlop != cols - 1:
        returned.append(SegmentedImage(segmented_img.x_coord + utolso_feher_oszlop + 1, segmented_img.y_coord,
                                       cols - (utolso_feher_oszlop + 1), segmented_img.height,
                                       segmented_img.img[:, utolso_feher_oszlop + 1:cols], segmented_img.rownum))

    if not returned:
        returned.append(segmented_img)

    return returned
