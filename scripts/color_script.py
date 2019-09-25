"""
Script for rapid false coloring large datasets

Rob Serafin
09/12/2019
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
    parser.add_argument("fname", help='imaris file')
    parser.add_argument("alpha", type = float, help='imaris file')
    parser.add_argument("savefolder",help='imaris file')
    args = parser.parse_args()

    filename = os.path.basename(args.filepath)

    #make save directory
    save_dir = args.savefolder + os.sep + "RGB"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    #load data
    f = h5.File(os.path.abspath(args.filename),'r')

    #downsampled data for flat fielding
    nuclei_ds = f['/t00000/s00/4/cells']
    cyto_ds = f['/t00000/s01/4/cells']

    #calculate flat field
    M_nuc,bkg_nuc = fc.getFlatField(nuclei_ds)
    M_cyt,bkg_cyt = fc.getFlatField(cyto_ds)

    dataQueue = mp.Queue()
    save_thread = mp.Process(target=saveProcess,args=dataQueue)

    #create reference to full res data
    nuclei_hires = f['/t00000/s00/0/cells']
    cyto_hires = f['/t00000/s01/0/cells']

    #block size for Image data
    tileSize = 256

    #settings for RGB conversion
    nuclei_RGBsettings = [0.65, 0.85, 0.35]
    cyto_RGB_settings = [0.05, 1.00, 0.544]

    for k in range(nuclei_ds.shape[1]*16):

        #get image data from both channels in blocks that are multiples of tileSize
        #subtract background and reset values > 0 and < 2**16
        nuclei = nuclei_hires[0:tileSize*M_nuc.shape[0],k,0:tileSize*M_nuc.shape[2]].astype(float)
        nuclei -= bkg_nuc
        nuclei = numpy.clip(nuclei,0,65535)

        cyto = cyto_hires[0:tileSize*M_cyt.shape[0],k,0:tileSize*M_cyt.shape[2]].astype(float)
        cyto -= 3*bkg_cyt
        cyto = numpy.clip(cyto,0,65535)

        x0 = numpy.floor(k/tileSize)
        x1 = numpy.ceil(k/tileSize)
        x = k/tileSize

        #sharpen images
        nuclei = fc.sharpenImage(nuclei)
        cyto = fc.sharpenImage(cyto)

        #get background block
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

        C_nuc = ndimage.interpolation.zoom(C_nuc, order = 1, mode = 'nearest')
        # C_nuc = 4.72*ndimage.filters.gaussian_filter(C_nuc,100)

        C_cyt = ndimage.interpolation.zoom(C_cyt, order = 1, mode = 'nearest')
        # C_cyt = 4.72*ndimage.filters.gaussian_filter(C_cyt,100)

        RGB_image = fc.rapidFalseColor(nuclei,cyto,nuclei_RGBsettings,cyto_RGB_settings,
                                        nuc_normfactor = C_nuc, cyto_normfactor = C_cyt,
                                        run_normalization = True)




















