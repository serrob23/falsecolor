import argparse
import tables as tb
from skimage.external import tifffile
from skimage.exposure import equalize_adapthist
from skimage.filters  import gaussian
import os
import h5py as h5
import numpy as np
import multiprocessing as mp
from functools import partial
import scipy.ndimage
from scipy import signal
import FalseColor.Color as fc

def main():

    ## INPUTS
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help='imaris file')
    parser.add_argument("fname", help='imaris file')
    parser.add_argument("alpha", type = float, help='imaris file')
    args = parser.parse_args()

    alpha = args.alpha
    tileSize = 256 # hardcoded for now

    ## SETUP PATHS
    file_path = os.path.abspath(args.filename)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)

    ## MAKE example FOLDER FOR TIFFS
    if not os.path.exists(dir_name + '\\example'):
        os.mkdir(dir_name + '\\example')

    ## READ DOWNSAMPLED DATA TO CALCULATE NORMILIZATION FACTORS
    f = h5.File(file_path,'r')

    channel_0_ds = f['/t00000/s00/4/cells']
    channel_1_ds = f['/t00000/s01/4/cells']

    ## EMPIRICALLY CALCULATE MEAN AND BKG INTENSITIES
    # MEAN IS 90% MAX INTENSITY
    # BKG IS MEAN / 5
    mean_0 = np.sort(channel_0_ds, axis = None)
    mean_0 = mean_0[np.where(mean_0 > 50)]
    mean_0 = mean_0[int(np.round(mean_0.shape[0]*0.95))]
    mean_1 = np.sort(channel_1_ds, axis = None)
    mean_1 = mean_1[np.where(mean_1 > 50)]
    mean_1 = mean_1[int(np.round(mean_1.shape[0]*0.95))]

    bkg_0 = mean_0 / 5
    bkg_1 = mean_1 / 5

    print('channel 0 background: ' + str(bkg_0))
    print('channel 0 background: ' + str(bkg_1))

    ## DETERMINE INDICES FOR CHUNKING DOWNSAMPLED DATA INTO TILESIZE
    # CLIP SO THAT XYZ EXTENTS ARE A MULTIPLE OF TILESIZE
    rows_max = int(np.floor(channel_0_ds.shape[0]/16)*16)
    cols_max = int(np.floor(channel_0_ds.shape[2]/16)*16)
    stacks_max = int(np.floor(channel_0_ds.shape[1]/16)*16)

    rows = np.arange(0, rows_max+int(tileSize/16), int(tileSize/16))
    cols = np.arange(0, cols_max+int(tileSize/16), int(tileSize/16))
    stacks = np.arange(0, stacks_max+int(tileSize/16), int(tileSize/16))

    M_0 = np.zeros((len(rows)-1, len(stacks)-1, len(cols)-1), dtype = float)
    M_1 = np.zeros((len(rows)-1, len(stacks)-1, len(cols)-1), dtype = float)

    ## CALCULATE NORMILIZATION MATRIX
    # FOR EACH TILE IN XYZ CALCULATE THE MEDIAN INTENSITY OF PIXELS > BKG
    # IF NO PIXELS ARE > BKG, SET VALUE TO MEAN 
    for i in range(1,len(rows)):
        for j in range(1,len(stacks)):
            for k in range(1,len(cols)):

                ROI_0 = channel_0_ds[rows[i-1]:rows[i], stacks[j-1]:stacks[j], cols[k-1]:cols[k]]
                
                fkg_ind = np.where(ROI_0 > bkg_0)
                if fkg_ind[0].size==0:
                    Mtemp = mean_0
                else:
                    Mtemp = np.median(ROI_0[fkg_ind])
                M_0[i-1, j-1, k-1] = Mtemp + M_0[i-1, j-1, k-1]

                ROI_1 = channel_1_ds[rows[i-1]:rows[i], stacks[j-1]:stacks[j], cols[k-1]:cols[k]]

                fkg_ind = np.where(ROI_1 > bkg_1)
                if fkg_ind[0].size==0:
                    Mtemp = mean_1
                else:
                    Mtemp = np.median(ROI_1[fkg_ind])
                M_1[i-1, j-1, k-1] = Mtemp + M_1[i-1, j-1, k-1]


    ## SET example VALUES FOR H&E CHANNELS
    beta2 = 0.05;
    beta4 = 1.00;
    beta6 = 0.544;

    beta1 = 0.65;
    beta3 = 0.85;
    beta5 = 0.35;

    ## LOOP THROUGH Z-PLANES AND FALSECOLOR
    for k in range(0, channel_0_ds.shape[1]*16):

        print(k)

        # ONLY GRAB XY EXTENTS THAT ARE MULTIPLES OF TILESIZE
        temp_0 = f['/t00000/s00/0/cells'][0:tileSize*M_0.shape[0],k,0:tileSize*M_0.shape[2]]
        temp_1 = f['/t00000/s01/0/cells'][0:tileSize*M_0.shape[0],k,0:tileSize*M_0.shape[2]]
        temp_0 = temp_0.astype(float)
        temp_1 = temp_1.astype(float)

        ind0 = np.where(temp_0 < 0)
        ind1 = np.where(temp_1 < 0)
        temp_0[ind0] = 32768 - (-32768 - temp_0[ind0])
        temp_1[ind1] = 32768 - (-32768 - temp_1[ind1])

        ## SHARPENING
        kernel1 = np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1],
        ])

        kernel2 = np.array([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1],
        ])

        temp_0 = temp_0 + alpha*np.sqrt(signal.convolve2d(temp_0,kernel1,mode='same')**2 + signal.convolve2d(temp_0,kernel2,mode='same')**2)
        temp_1 = temp_1 + alpha*np.sqrt(signal.convolve2d(temp_1,kernel1,mode='same')**2 + signal.convolve2d(temp_1,kernel2,mode='same')**2)

        tifffile.imsave(dir_name + '\\example\\T0_' + '{:0>6d}'.format(k) + '.tif', temp_0.astype('uint16'))
        tifffile.imsave(dir_name + '\\example\\T1_' + '{:0>6d}'.format(k) + '.tif', temp_1.astype('uint16'))

        # INTERPOLATE THE NORMALIZATION MATRIX FOR THE GIVEN Z-PLANE
        # ALSO INTERPOLATE ITS RESOLUTION UP TO MATCH THE RAW DATA
        if k < int(M_0.shape[1]*256-tileSize):
            if k < int(tileSize/2):
                C_0 = M_0[:,0, :]
                C_0 = scipy.ndimage.interpolation.zoom(C_0, tileSize, order = 1, mode = 'nearest')

                C_1 = M_1[:,0, :]
                C_1 = scipy.ndimage.interpolation.zoom(C_1, tileSize, order = 1, mode = 'nearest')
            else:
                x0 = np.floor(k/tileSize)
                x1 = np.ceil(k/tileSize)
                x = k/tileSize
                y0 = M_0[:,int(x0),:]
                y1 = M_0[:,int(x1),:]

                if x1 == x0:
                    C_0 = y1
                else:
                    C_0 = y0 + (x - x0)*(y1 - y0)/(x1 - x0)
                C_0 = scipy.ndimage.interpolation.zoom(C_0, tileSize, order = 1, mode = 'nearest')

                y0 = M_1[:,int(x0),:]
                y1 = M_1[:,int(x1),:]

                if x1 == x0:
                    C_1 = y1
                else:
                    C_1 = y0 + (x - x0)*(y1 - y0)/(x1 - x0)
                C_1 = scipy.ndimage.interpolation.zoom(C_1, tileSize, order = 1, mode = 'nearest')
        else:
            C_0 = M_0[:,M_0.shape[1]-1, :]
            C_0 = scipy.ndimage.interpolation.zoom(C_0, tileSize, order = 1, mode = 'nearest')

            C_1 = M_1[:,M_1.shape[1]-1, :]
            C_1 = scipy.ndimage.interpolation.zoom(C_1, tileSize, order = 1, mode = 'nearest')

        C_0 = gaussian(C_0, sigma = 100)
        C_1 = gaussian(C_1, sigma = 100)

        # CLIP OFF THE BACKGROUND TO 0
        np.clip(temp_0,bkg_0, 65535) - bkg_0
        np.clip(temp_1,3*bkg_1, 65535) - 3*bkg_1

        # FALSE COLOR IMAGES WITH BEER LAMBERT LAW
        im = np.zeros((temp_0.shape[0], temp_0.shape[1], 3), dtype = float)
        im[:,:,0] = np.multiply(np.exp(np.divide(-temp_0*beta1,(4.72*C_0))), np.exp(np.divide(-temp_1*beta2,(4.72*C_1))))
        im[:,:,1] = np.multiply(np.exp(np.divide(-temp_0*beta3,(4.72*C_0))), np.exp(np.divide(-temp_1*beta4,(4.72*C_1))))
        im[:,:,2] = np.multiply(np.exp(np.divide(-temp_0*beta5,(4.72*C_0))), np.exp(np.divide(-temp_1*beta6,(4.72*C_1))))

        # CONVERT TO 8-BIT RGB AND SAVE FILE
        im = im*255
        im = im.astype('uint8')
        tifffile.imsave(dir_name + '\\' + args.fname + '\\' + '{:0>6d}'.format(k) + '.tif', im)

    f.close()

if __name__ == '__main__':
    print("H5 to RGB")
    main()
