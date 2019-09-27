"""
Script for rapid false coloring large datasets

Rob Serafin
09/26/2019
"""

import os
import glob
import FalseColor.Color as fc
from FalseColor.SaveThread import saveProcess
import numpy 
from scipy import ndimage
import skimage.filters as filt
import cv2
import copy
import multiprocessing as mp
import argparse
import h5py as h5

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help='imaris file')
    parser.add_argument("filename", help='imaris file')
    parser.add_argument("alpha", type = float, help='imaris file')
    parser.add_argument("savefolder",help='imaris file')
    parser.add_argument("format",type=str,help='imagris file')
    parser.add_argument("stop_k",type=int,help='imagris file')
    args = parser.parse_args()


    #get path info
    filename = args.filename
    filepath = args.filepath
    save_dir = args.savefolder

    stop_k = args.stop_k

    #load data
    datapath = filepath + os.sep + filename
    print(datapath)
    f = h5.File(datapath,'r')
    print(f.keys())

    #downsampled data for flat fielding
    nuclei_ds = f['/t00000/s00/4/cells']
    cyto_ds = f['/t00000/s01/4/cells']

    #calculate flat field
    M_nuc,bkg_nuc = fc.getFlatField(nuclei_ds)
    M_cyt,bkg_cyt = fc.getFlatField(cyto_ds)

    dataQueue = mp.Queue()
    save_thread = mp.Process(target=saveProcess,args=[dataQueue])
    save_thread.start()

    #create reference to full res data
    nuclei_hires = f['/t00000/s00/0/cells']
    cyto_hires = f['/t00000/s01/0/cells']

    #block size for Image data
    tileSize = 256

    #settings for RGB conversion
    nuclei_RGBsettings = [0.54, 1.0, 0.35]
    cyto_RGB_settings = [0.3, 1.00, 0.86]

    for k in range(nuclei_ds.shape[1]*16):

        if k % 2 == 0:
            print('on section: ',k)

        if k == stop_k:
            break
        else:

            #get image data from both channels in blocks that are multiples of tileSize
            #subtract background and reset values > 0 and < 2**16
            print('line 80')
            nuclei = nuclei_hires[0:tileSize*M_nuc.shape[0],500+k,0:tileSize*M_nuc.shape[2]].astype(float)
            nuclei -= bkg_nuc
            nuclei = numpy.clip(nuclei,0,65535)

            print('line 85')
            cyto = cyto_hires[0:tileSize*M_cyt.shape[0],500+k,0:tileSize*M_cyt.shape[2]].astype(float)
            cyto -= bkg_cyt
            cyto = numpy.clip(cyto,0,65535)

            print('line 90')
            x0 = numpy.floor(k/tileSize)
            x1 = numpy.ceil(k/tileSize)
            x = k/tileSize

            # sharpen images
            print('sharpening')
            nuclei = fc.sharpenImage(nuclei)
            cyto = fc.sharpenImage(cyto)

            #get background block
            print('background block')
            if k < int(M_nuc.shape[1]*tileSize-tileSize):
                if k < int(tileSize/2):
                    C_nuc = M_nuc[:,0,:]
                    C_cyt = M_cyt[:,0,:]

                elif x0==x1:
                    #TODO: ask AK about x0 vs x1 in C_nuc
                    C_nuc = M_nuc[:,int(x1),:]
                    C_cyt = M_cyt[:,int(x1),:]
                else:
                    C_nuc = M_nuc[:,int(x0),:]
                    C_cyt = M_cyt[:,ing(x1),:]
                    diff = C_cyt - C_nuc

                    C_nuc += (x-x0)*diff/(x1-x0)
                    C_cyt = copy.deepcopy(C_nuc)
            else:
                C_nuc = M_nuc[:,M_nuc.shape[1]-1, :]
                C_cyt = M_cyt[:,M_cyt.shape[1]-1, :]

            print('interpolating')
            C_nuc = ndimage.interpolation.zoom(C_nuc, tileSize, order = 1, mode = 'nearest')
            # C_nuc = 4.72*ndimage.filters.gaussian_filter(C_nuc,100)

            C_cyt = ndimage.interpolation.zoom(C_cyt, tileSize, order = 1, mode = 'nearest')
            # C_cyt = 4.72*ndimage.filters.gaussian_filter(C_cyt,100)
            print('RGB')
            RGB_image = fc.rapidFalseColor(nuclei,cyto,nuclei_RGBsettings,cyto_RGB_settings,
                                            nuc_normfactor = C_nuc, cyto_normfactor = C_cyt,
                                            run_normalization = True)


            #append data to queue
            save_file = '{:0>6d}'.format(k) + '.tif'
            message = [filepath,save_dir,save_file,RGB_image,None]
            dataQueue.put(message)


    #stop data queue
    stop_message = [None,None,None,None,'stop']
    dataQueue.put(stop_message)
    save_thread.join()
    f.close()

if __name__ == '__main__':
    print('False Coloring')
    main()




















