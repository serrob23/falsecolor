import argparse
import tables as tb
from skimage.external import tifffile as tiff
import os.path
import h5py as h5
import numpy as np
from tqdm import tqdm
import cv2 as cv
import glob
import timeit
import multiprocessing as mp
from functools import partial

def false_color_images(folder, files1, files2, k):
    
    beta2 = 0.05
    beta4 = 1.00
    beta6 = 0.544

    beta1 = 0.65
    beta3 = 0.85
    beta5 = 0.35

    temp_0 = cv2.imread(files1[k],cv2.IMREAD_ANYDEPTH)
    temp_1 = cv2.imread(files2[k],cv2.IMREAD_ANYDEPTH)
    temp_0 = temp_0.astype(float)
    temp_1 = temp_1.astype(float)
    temp_0 -= 50
    temp_1 -= 50
    temp_0[np.where(temp_0 < 0)] = 0
    temp_1[np.where(temp_1 < 0)] = 0
    #temp_0 = cv.resize(temp_0,(int(temp_0.shape[0]*1.414), temp_0.shape[1]), interpolation = cv.INTER_CUBIC)
    #temp_1 = cv.resize(temp_1,(int(temp_1.shape[0]*1.414), temp_1.shape[1]), interpolation = cv.INTER_CUBIC)
    im = np.zeros((temp_0.shape[0], temp_0.shape[1], 3))
    im[:,:,0] = np.multiply(np.exp(-temp_0*beta1/1200), np.exp(-temp_1*beta2/1200))
    im[:,:,1] = np.multiply(np.exp(-temp_0*beta3/1200), np.exp(-temp_1*beta4/1200))
    im[:,:,2] = np.multiply(np.exp(-temp_0*beta5/1200), np.exp(-temp_1*beta6/1200))
    im = im*255
    im = im.astype('uint8')
    oldname = files1[k]
    #newname = oldname.replace('CH000001', '')
    newname = oldname.replace(folder, folder + '_rgb')
    tifffile.imsave(newname, im)

    return

def main():

    tqdm.monitor_interval = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help='folder of TIF files')
    args = parser.parse_args()

    cwd = os.getcwd()
    path = args.folder
    files1 = glob.glob(path + '\*_CH000001.tif')
    files2 = glob.glob(path + '\*_CH000002.tif')

    #false-color images

    print('Coloring images...')

    start = timeit.default_timer()
	num_cores = mp.cpu_count()
    pool = mp.Pool(processes = num_cores)
    func_p = partial(false_color_images, path, files1, files2)
    img_temp = pool.map(func_p, (z for z in xrange(0, len(files1))))

    stop = timeit.default_timer()
    print('Total time: {0}s'.format(stop-start))

if __name__ == '__main__':
    main()
