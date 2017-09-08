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
                'v':self.hist(channel=2),}
        return diff

    def calHash(self, image=None, size=(8, 8)):
        imr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imr = cv2.resize(imr, size)
        im_cp = imr > np.mean(imr)
        im_fl = im_cp.ravel().astype(int)
        return im_fl

    def hamming_distance(self, pair=None):

        differ = np.abs(pair[0] - pair[1])
        return np.sum(differ)


    def ahash(self):
        pass

    def phash(self):
        pass

    def dhash(self):
        pass


if __name__ == '__main__':

    file = ('1.png', '4.png')
    comp = imagesimi(pair=file)
    print(comp.hsvhist())
    a_hash = comp.calHash(image=comp.ia, size=(8, 8))
    b_hash = comp.calHash(image=comp.ib, size=(8, 8))
    diff_hash = comp.hamming_distance(pair=(a_hash, b_hash))
    print(a_hash, b_hash, diff_hash)