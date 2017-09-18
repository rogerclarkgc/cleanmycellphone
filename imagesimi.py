import cv2
import numpy as np
from matplotlib import pyplot as plt

class imagefeature(object):
    """
    use hist similarity, hash features to compare the similarity between two pictures
    """

    def __init__(self, img=None):
        """
        initiating the class
        :param img: two picture's file name, construct like (a, b)
        """
        self.img = cv2.imread(img)
        if  not (isinstance(self.img, np.ndarray)):
            raise RuntimeError('no picture.')


    def hist(self, size=(256, 256), channel=0):
        """
        calculate the similarity between tow pic's color histgram
        :param size: resize the picture
        :param channel:color chanel, from 0 to 2, represent H, S, V
        :return:a value represent similarity between 0 and 1
        """
        ra_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        ra_hsv = cv2.resize(ra_hsv, size)
        if channel == 0:
            hista = cv2.calcHist([ra_hsv], [channel], None, [181], [0, 181])
        else:
            hista = cv2.calcHist([ra_hsv], [channel], None, [256], [0, 256])
        hista = np.array([w[0] for w in hista])
        return hista

    def calHash(self, image=None):
        #imr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #imr = cv2.resize(imr, size)
        im_cp = image > np.mean(image)
        im_fl = im_cp.ravel().astype(int)
        return im_fl

    def ahash(self, size=(8, 8)):

        im1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        im1 = cv2.resize(im1, size)
        img_ahash = self.calHash(image=im1)
        return img_ahash

    def phash(self, size=(32, 32)):

        im1_re = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        im1_re = cv2.resize(im1_re, size, interpolation=cv2.INTER_CUBIC)

        #dct1 = cv2.dct(cv2.dct(im1_re.astype(np.float32)), cv2.DCT_INVERSE, flags=1)
        dct1 = cv2.dct(im1_re.astype(np.float32))
        cv2.imwrite('dct1.jpg', dct1)
        dct1_roi = dct1[:8, :8]
        #dct1_roi = cv2.resize(dct1, (8, 8))

        img_phash = self.calHash(dct1_roi)
        return img_phash


    def ColorLayout(self, split=(8, 8)):
        # split the image in m*n squads
        rowstep_a = round(self.img.shape[0]/split[0])
        colstep_a = round(self.img.shape[1]/split[1])
        row_a = np.arange(0, self.img.shape[0], rowstep_a)
        col_a = np.arange(0, self.img.shape[1], colstep_a)
        rep_a = np.zeros(self.img.shape)
        mat_a = np.zeros(split+(3, ))

        # calculate each pictures' represent color in (m, n) split squads, use mean of RGB value
        for ix, x in enumerate(row_a):
            for iy, y in enumerate(col_a):
                rep_a[x:x+rowstep_a, y:y+colstep_a] = np.mean(self.img[x:x+rowstep_a, y:y+colstep_a], axis=(0, 1))
                try:
                    mat_a[ix, iy] = np.mean(self.img[x:x+rowstep_a, y:y+colstep_a], axis=(0, 1))
                except IndexError:
                    pass

        #cv2.imwrite('cloa.jpg', rep_a)

        return {'mosaic':rep_a, 'small':mat_a}

    def cldct(self, cl = None):

        small_mat = cl['small'].astype(np.uint8)
        mat_yuv = cv2.cvtColor(small_mat, cv2.COLOR_BGR2YCR_CB)
        dct_y = cv2.dct(mat_yuv[:, :, 0].astype(np.float32))
        dct_cr = cv2.dct(mat_yuv[:, :, 1].astype(np.float32))
        dct_cb = cv2.dct(mat_yuv[:, :, 2].astype(np.float32))
        return (dct_y, dct_cr, dct_cb)


class imgcompare(object):
    """

    """
    def __init__(self, featurepool=None, request=None):

        self.pool = featurepool
        self.req = request
        self.lenreq = len(self.req)
        self.poolnum = len(self.pool)

    def hamming_distance(self, pair=None):
        differ = np.abs(pair[0] - pair[1])
        return np.sum(differ)

    def comparehist(self):
        compare = []
        for index, item in enumerate(self.pool):
            maxab = np.array([max(item[i], self.req[i]) for i in range(self.lenreq)])
            term1 = np.nan_to_num(np.abs(item-self.req)/maxab)
            differ = np.mean(1-term1)
            comp = (index, differ)
            compare.append(comp)

        max_simi = max(compare, key=lambda x:x[1])
        min_simi = min(compare, key=lambda x:x[1])

        return {'all': compare,
                'max': max_simi,
                'min': min_simi}

    def comparehash(self):
        compare = []
        for index, item in enumerate(self.pool):
            diff_hash = self.hamming_distance(pair=(self.req, item))
            comp = (index, diff_hash)
            compare.append(comp)

        max_simi = max(compare, key=lambda x:x[1])
        min_simi = min(compare, key=lambda x:x[1])

        return {'all': compare,
                'max': max_simi,
                'min': min_simi}




if __name__ == '__main__':

    from matplotlib import pyplot as plt

    index = np.arange(0, 1, 0.05)
    index = list(map(lambda x:'s_{}.jpg'.format(str(x)), index[1:]))
    index.append('s_1.jpg')
    #req = 's_0.01.jpg'

    histpool = [imagefeature(image).hist() for image in index]
    ahashpool = [imagefeature(image).ahash() for image in index]
    phashpool = [imagefeature(image).phash() for image in index]
    clpool = [imagefeature(image).ColorLayout() for image in index]

    compare_hist = imgcompare(featurepool=histpool, request=histpool[0]).comparehist()
    compare_ahash = imgcompare(featurepool=ahashpool, request=ahashpool[0]).comparehash()
    compare_phash = imgcompare(featurepool=phashpool, request=phashpool[0]).comparehash()

    print(compare_hist['max'], compare_ahash['min'], compare_phash['min'])
    #print(clpool[0])
    #cv2.imwrite('clpool_0.jpg', clpool[0]['mosaic'])






