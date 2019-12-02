"""
#===============================================================================
# 
#  License: GPL
#
#
#  Copyright (c) 2019 Rob Serafin, Liu Lab, 
#  The University of Washington Department of Mechanical Engineering  
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License 2
#  as published by the Free Software Foundation.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
# 
#===============================================================================

Rob Serafin
11/20/2019

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
    settings_dict = fc.getDefaultRGBSettings()
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
            nuclei = nuclei_hires[0:tileSize*M_nuc.shape[0],k,0:tileSize*M_nuc.shape[2]]
            nuclei = nuclei.astype(float)
            nuclei -= 0.5*bkg_nuc
            nuclei = numpy.clip(nuclei,0,65535)
            print('read time nuclei', time.time()-t_nuc)

            t_cyt = time.time()
            cyto = cyto_hires[0:tileSize*M_cyt.shape[0],k,0:tileSize*M_cyt.shape[2]]
            cyto = cyto.astype(float)
            cyto -= 3*bkg_cyt
            cyto = numpy.clip(cyto,0,65535)
            print('read time cyto', time.time() - t_cyt)

            # sharpen images
            print('sharpening')
            nuclei = fc.sharpenImage(nuclei)
            cyto = fc.sharpenImage(cyto)

            #interpolate downsampled images to full res size to use as flat fielding mask
            C_nuc, C_cyt = fc.interpolateDS(M_nuc, M_cyt, k)

            print('False Coloring')

            #Execute false coloring method
            RGB_image = fc.rapidFalseColor(nuclei,cyto,nuclei_RGBsettings,cyto_RGBsettings,
                                            nuc_normfactor = 1.0*C_nuc, 
                                            cyto_normfactor = 3.72*C_cyt,
                                            run_normalization = True)

            #append data to queue
            save_file = '{:0>6d}'.format(k) + args.format
            message = [filepath,save_dir,save_file,RGB_image,None]
            t0 = time.time()
            dataQueue.put(message)

            nuclei = None
            cyto = None
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


