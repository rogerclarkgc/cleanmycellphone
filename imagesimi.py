import cv2
import numpy as np
from matplotlib import pyplot as plt

class imagesimi(object):
    """
    use hist similarity, hash features to compare the similarity between two pictures
    """

    def __init__(self, pair=None):
        """
        initiating the class
        :param pair: two picture's file name, construct like (a, b)
        """
        self.ia = cv2.imread(pair[0])
        self.ib = cv2.imread(pair[1])
        if  not (isinstance(self.ia, np.ndarray) and isinstance(self.ib, np.ndarray)):
            raise RuntimeError('no picture.')


    def hist(self, size=(256, 256), channel=0):
        """
        calculate the similarity between tow pic's color histgram
        :param size: resize the picture
        :param channel:color chanel, from 0 to 2, represent H, S, V
        :return:a value represent similarity between 0 and 1
        """
        ra_hsv = cv2.cvtColor(self.ia, cv2.COLOR_BGR2HSV)
        rb_hsv = cv2.cvtColor(self.ib, cv2.COLOR_BGR2HSV)
        ra_hsv = cv2.resize(ra_hsv, size)
        rb_hsv = cv2.resize(rb_hsv, size)
        if channel == 0:
            hista = cv2.calcHist([ra_hsv], [channel], None, [181], [0, 181])
            histb = cv2.calcHist([rb_hsv], [channel], None, [181], [0, 181])
        else:
            hista = cv2.calcHist([ra_hsv], [channel], None, [256], [0, 256])
            histb = cv2.calcHist([rb_hsv], [channel], None, [256], [0, 256])
        hista = np.array([w[0] for w in hista])
        histb = np.array([w[0] for w in histb])
        maxab = np.array([max(hista[i], histb[i]) for i in range(len(hista))])
        try:
            term1 = np.nan_to_num(np.abs(hista-histb)/maxab)
        except RuntimeWarning:
            pass
        differ = np.mean(1-term1)
        return differ

    def hsvhist(self):
        """
        calculate all chanel in one time
        :return: a dict object, contain all channels ' similarity
        """
        diff = {'h':self.hist(channel=0),
                's':self.hist(channel=1),
                'v':self.hist(channel=2)}
        return diff

    def calHash(self, image=None):
        #imr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #imr = cv2.resize(imr, size)
        im_cp = image > np.mean(image)
        im_fl = im_cp.ravel().astype(int)
        return im_fl

    def hamming_distance(self, pair=None):

        differ = np.abs(pair[0] - pair[1])
        return np.sum(differ)


    def ahash(self, size=(8, 8)):

        im1 = cv2.cvtColor(self.ia, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(self.ib, cv2.COLOR_BGR2GRAY)
        im1 = cv2.resize(im1, size)
        im2 = cv2.resize(im2, size)
        a_hash = self.calHash(image=im1)
        b_hash = self.calHash(image=im2)
        diff_hash = self.hamming_distance(pair=(a_hash, b_hash))
        return diff_hash

    def phash(self, size=(32, 32)):

        im1_re = cv2.cvtColor(self.ia, cv2.COLOR_BGR2GRAY)
        im1_re = cv2.resize(im1_re, size, interpolation=cv2.INTER_CUBIC)
        im2_re = cv2.cvtColor(self.ib, cv2.COLOR_BGR2GRAY)
        im2_re = cv2.resize(im2_re, size, interpolation=cv2.INTER_CUBIC)

        dct1 = cv2.dct(cv2.dct(im1_re.astype(np.float32)))
        dct2 = cv2.dct(cv2.dct(im2_re.astype(np.float32)))

        dct1_roi = dct1[:8, :8]
        #dct1_roi = cv2.resize(dct1, (8, 8))
        dct2_roi = dct2[:8, :8]
        #dct2_roi = cv2.resize(dct2, (8, 8))

        hash1 = self.calHash(dct1_roi)
        hash2 = self.calHash(dct2_roi)

        diff_hash = self.hamming_distance(pair=(hash1, hash2))
        return diff_hash


    def dhash(self):
        pass


if __name__ == '__main__':

    file = ('3.jpg', '4.jpg')
    comp = imagesimi(pair=file)
    print(comp.hsvhist())
    print(comp.ahash(size=(8, 8)))
    print(comp.phash())
