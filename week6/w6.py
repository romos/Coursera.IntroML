import numpy as np
import skimage
import skimage.io

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

__author__ = 'oderor'


def readdataNP(fin):
    # returns n x m x 3 ndarray.
    # n x m - image size; 3 - due to RGB code
    return skimage.io.imread(fin)


def writefile(str, filename):
    fout = open(filename, 'w')
    print(filename, ':', str)
    print(str, file=fout, end="")
    fout.close()


def PSNR(orig, approx, MAXI = 255):
    # First, compute MSE
    (m, n) = orig.shape
    mse = np.sum(np.square(1.0*orig - 1.0*approx)) / (m * n)
    # mse = mean_squared_error(orig, approx)

    # Then, compute PSNR itself
    psnr = 20.0 * np.log10(MAXI) - 10.0 * np.log10(mse)
    return psnr


def func(image):
    # Transform all pixel values into [0..1] range
    image_orig = image
    image = skimage.img_as_float(image_orig)

    # Create object-feature matrix. Each pixel is an object. Its features are R,G,B values
    r = image[:, :, 0].ravel()
    g = image[:, :, 1].ravel()
    b = image[:, :, 2].ravel()
    X = np.column_stack((r, g, b))

    # Run KMeans over the object-feature matrix X
    # Try multiple numbers of clusters
    PSNR_EPSILON = 20
    C_PSNR_EPSILON = -1
    CLUSTERS = range(1, 21)
    for c in CLUSTERS:
        kmestimator = KMeans(n_clusters=c, init='k-means++', random_state=241)
        kmestimator.fit(X)

        # Set pixels in the same cluster to their MEDIAN value
        X_median = X.copy()
        for i in range(kmestimator.n_clusters):
            X_median[kmestimator.labels_ == i] = np.median(X[kmestimator.labels_ == i], axis=0)
        # Set pixels in the same cluster to their MEAN value
        X_mean = X.copy()
        for i in range(kmestimator.n_clusters):
            X_mean[kmestimator.labels_ == i] = np.mean(X[kmestimator.labels_ == i], axis=0)

        # fig = plt.figure()
        # a = fig.add_subplot(1, 2, 1)
        # a.set_title('N_CLUSTERS=%d, FILL=%s' % (c, 'MEDIAN'))
        # plt.imshow(np.reshape(X_median, image_orig.shape))
        # a = fig.add_subplot(1, 2, 2)
        # a.set_title('N_CLUSTERS=%d, FILL=%s' % (c, 'MEAN'))
        # plt.imshow(np.reshape(X_mean, image_orig.shape))

        # Compute PSNR error:
        psnr_median = PSNR(X,X_median,MAXI=1)
        psnr_mean = PSNR(X,X_mean,MAXI=1)

        print('CLUSTER: %2d | PSNR_MEAN: %0.5f | PSNR_MEDIAN: %0.5f' % (c,psnr_mean,psnr_median))

        if psnr_mean > PSNR_EPSILON:
            C_PSNR_EPSILON = c
            break

    # plt.show()
    return C_PSNR_EPSILON


def main():
    # ------------------------------------------------------
    # Part 1. Image processing. Reduced colors
    # ------------------------------------------------------
    fin = 'parrots.jpg'
    image = readdataNP(fin)
    print('Image processing...')
    c = func(image)
    s = '%d' % (c)
    writefile(s, fin + ".out")
    print('Completed')


if __name__ == '__main__':
    main()
