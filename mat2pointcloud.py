'''
This module converts depth map captured with kinect/Astra depth camera to point cloud
the input to the system is a set of depth images, this module stacks the depth images
and creates a point cloud from the median value at a coordinate

dirpath is the first cmd arg specifying the path to look for images
camtype specifying the type of camera is the second arg, either kinect or Astra
'''
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from matplotlib import cm
from PIL import Image
import cv2
import sys
import sklearn
from datetime import datetime


def readDndDumpPC(path, camtype):
    print(path, camtype)

    # set the dimension of depth map as per camera type
    if camtype == 'kinect':
        dp = np.empty([512, 424], dtype=float)
        filesuffix = r'\DepthFrame*.mat'
    elif camtype == 'astra':
        dp = np.empty([480, 640], dtype=float)
        filesuffix = r'\astra_dmpStream*mat'

    for filename in glob(path + filesuffix):

        if camtype == 'kinect':
            key = 'Dep'+filename.split('.')[0][-4:]+'_'
        elif camtype == 'astra':
            key = 'img'

        mat = scipy.io.loadmat(filename)
        tmp = np.asarray(mat[key])
        print(dp.shape, tmp.shape)
        dp = np.dstack((dp, mat[key]))

    if dp.size:
        print(dp.shape)
        print(np.unique(dp).shape)
        print(np.unique(dp))

    # replace the depth value with median value at that location
    dp = np.median(dp, axis=2)
    print(dp.shape)

    # crop as per requirement
    dp = dp[180:420, 270:470]
    dp = dp.astype(np.int32)

    # tune these params in bilateral Filtering
    def bilateralSmooth(dp):
        dp = dp.astype(np.uint8)
        blur = cv2.bilateralFilter(dp,9,75,75)
        cv2.imshow('orig', dp)
        cv2.waitKey(0)
        cv2.imshow('bilateralFilter', blur)
        cv2.waitKey(0)
        return blur

    filename = filename.split('.')[0]+str(datetime.now().microsecond)[-4:]
    print(filename)
    mask = saveSegmMask(dp, filename+'_mask.jpg')
    cmp = cm.ScalarMappable(cmap='Spectral').to_rgba(dp, bytes=True)

    # from colormap array to PIL and save as JPG
    image = Image.fromarray(cmp[:, :, :3])
    image.save(filename+'_depth.jpg')

    with open(filename + '.xyz', 'w+') as pcfile:
        for i in range(dp.shape[0]):
            for j in range(dp.shape[1]):
                if mask[i, j] == 0:
                    pcfile.write(str(i) + " " + str(j) + " " + str(dp[i][j] % 1200) + " "
                                + str(cmp[i][j][0]) + " " + str(cmp[i][j][1]) + " " + str(cmp[i][j][2])
                                + "\n")

# plot the depth color map created
def plotDepth(dp):
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    dp = dp.astype(np.int32)
    dp = np.subtract(dp, 600)
    dp[np.where(dp < 0)] = 0

    # crop the section to plot, change the colormap to be used here
    plt.imshow(dp.T[20:250, 20:250], cmap='Spectral')
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()


# do a OTSU adaptive threshold followed by segmentation
def saveSegmMask(dp, fname):
    dp = (255*dp/np.max(dp)).astype('uint8')
    ret, mask = cv2.threshold(dp,0,np.max(dp),cv2.THRESH_OTSU)
    cv2.imshow('thres', mask)
    cv2.waitKey(0)
    cv2.imwrite(fname, mask)
    return mask


if __name__ == "__main__":
    try:
        readDndDumpPC(sys.argv[2], sys.argv[3])
    except:
        # call the default case with kinect type
        readDndDumpPC(sys.argv[2], 'kinect')