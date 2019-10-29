"""

MIT License

Copyright (c) [2019] [Robert Serafin]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



Script for rapid false coloring large datasets

Rob Serafin
09/26/2019
"""

import os
import FalseColor.Color as fc
from FalseColor.SaveThread import saveProcess
import numpy 
from scipy import ndimage
import copy
import multiprocessing as mp
import argparse
import h5py as h5
import time

def getDefaultRGBSettings():
    """returns empirically determined constants for nuclear/cyto channels

    Note: these settings currently only optimized for flat field method in
    rapidFalseColor
    beta2 = 0.05;
    beta4 = 1.00;
    beta6 = 0.544;

    beta1 = 0.65;
    beta3 = 0.85;
    beta5 = 0.35;
    """
    k_cyto = 1.0
    k_nuclei = 1.0
    nuclei_RGBsettings = [0.25*k_nuclei, 0.37*k_nuclei, 0.1*k_nuclei]
    cyto_RGB_settings = [0.05*k_cyto, 1.0*k_cyto, 0.54*k_cyto]

    settings_dict = {'nuclei':nuclei_RGBsettings,'cyto':cyto_RGB_settings}
    return settings_dict

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help='imaris file')
    parser.add_argument("filename", help='imaris file')
    parser.add_argument("alpha", type = float, help='imaris file')
    parser.add_argument("savefolder",help='imaris file')
    parser.add_argument("format",type=str,help='imagris file')
    parser.add_argument("start_k",type = int,help = 'imagris file')
    parser.add_argument("stop_k",type = int,help='imagris file')
    args = parser.parse_args()




    #get path info
    filename = args.filename
    filepath = args.filepath
    save_dir = args.savefolder


    #load data
    datapath = filepath + os.sep + filename

    f = h5.File(datapath,'r')

    #downsampled data for flat fielding
    nuclei_ds = f['/t00000/s00/4/cells']
    cyto_ds = f['/t00000/s01/4/cells']

    #indices to pseudo color, if stop_k = 0 the entire dataset from start_k on
    #will be false colored.
    start_k = args.start_k
    stop_k = args.stop_k

    if stop_k != 0:
        stop_k += start_k

    elif stop_k == 0:
        stop_k = nuclei_ds.shape[1]*16

    print(start_k,stop_k)

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
    settings_dict = getDefaultRGBSettings()
    nuclei_RGBsettings = settings_dict['nuclei']
    cyto_RGBsettings = settings_dict['cyto']
    print(nuclei_RGBsettings)
    print(cyto_RGBsettings)


    for k in range(start_k,stop_k):
        if k == stop_k:
            break
        else:

            t_start = time.time()

            print('on section: ',k)

            #get image data from both channels in blocks that are multiples of tileSize
            #subtract background and reset values > 0 and < 2**16
            print('Reading Data')
            t_nuc = time.time()
            nuclei = nuclei_hires[0:tileSize*M_nuc.shape[0],k,0:tileSize*M_nuc.shape[2]].astype(float)
            nuclei -= bkg_nuc
            nuclei = numpy.clip(nuclei,0,65535)
            print('read time nuclei', time.time()-t_nuc)

            t_cyt = time.time()
            cyto = cyto_hires[0:tileSize*M_cyt.shape[0],k,0:tileSize*M_cyt.shape[2]].astype(float)
            cyto -= bkg_cyt
            cyto = numpy.clip(cyto,0,65535)
            print('read time cyto', time.time() - t_cyt)

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
                    C_cyt = M_cyt[:,int(x1),:]
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

            print('False Coloring')
            RGB_image = fc.rapidFalseColor(nuclei,cyto,nuclei_RGBsettings,cyto_RGBsettings,
                                            nuc_normfactor = 2.72*C_nuc, 
                                            cyto_normfactor = 1.0*C_cyt,
                                            run_normalization = True)


            #append data to queue
            save_file = '{:0>6d}'.format(k) + args.format
            message = [filepath,save_dir,save_file,RGB_image,None]
            t0 = time.time()
            dataQueue.put(message)

            nuclei = None
            cyto = None
            print('transfer time:', time.time()-t0)
            print('runtime:', time.time() - t_start)


    #stop data queue
    stop_message = [None,None,None,None,'stop']
    dataQueue.put(stop_message)
    save_thread.join()
    f.close()

if __name__ == '__main__':
    t_overall = time.time()
    print('Starting False Color Script')
    main()
    print('total runtime: %s minutes' % ((time.time()-t_overall)/60))




















