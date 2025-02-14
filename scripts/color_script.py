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
3/25/2020

"""

import os
import falsecolor.coloring as fc
from falsecolor.savethread import saveProcess
import numpy
import argparse
import h5py as h5
import multiprocessing as mp
import time


def main():

    parser = argparse.ArgumentParser()

    # directory containing data to false color
    parser.add_argument("filepath", help='imaris file')

    # h5 data file
    parser.add_argument("filename", help='imaris file')

    # folder to save results
    parser.add_argument("savefolder", help='imaris file')

    # saved result format
    parser.add_argument("format", type=str, help='imagris file')

    # index to start/stop processing, skip_k is the stepsize interval
    parser.add_argument("start_k", type=int, help='imaris file')
    parser.add_argument("stop_k", type=int, help='imaris file')
    parser.add_argument("skip_k", type=int, nargs='?', default=1)

    # constants for coloring normalization and sharpening (alpha)
    parser.add_argument("Nuclei_Normfactor", type=float, nargs='?',
                        default=1.5)
    parser.add_argument("Cyto_Normfactor", type=float, nargs='?',
                        default=3.72)
    parser.add_argument("alpha", type=float, nargs='?',
                        default=0.5, help='imaris file')

    # get arguments
    args = parser.parse_args()

    # get path info
    filename = args.filename
    filepath = args.filepath
    save_dir = args.savefolder

    # load data
    datapath = os.path.join(filepath, filename)

    f = h5.File(datapath, 'r')

    # downsampled data for flat fielding
    nuclei_ds = f['/t00000/s00/4/cells']
    cyto_ds = f['/t00000/s01/4/cells']

    # indices to pseudo color, if stop_k = 0 the entire dataset from
    # start_k on will be processed, at intervals of skip_k
    # otherwise stop_k will be the total number of images processed
    # starting from start_k

    start_k = args.start_k
    stop_k = args.stop_k
    skip_k = args.skip_k

    # sharpening coefficient
    alpha = args.alpha

    # normalization coefficients
    nuc_norm_constant = args.Nuclei_Normfactor
    cyto_norm_constant = args.Cyto_Normfactor

    if stop_k != 0:
        stop_k += start_k

    elif stop_k == 0:
        stop_k = nuclei_ds.shape[1]*16

    print('Reading data from index:', start_k, 'to ', stop_k,
          'at stepsize = ', skip_k)

    # calculate flat field
    M_nuc = fc.getIntensityMap(nuclei_ds)
    M_cyto = fc.getIntensityMap(cyto_ds)

    bkg_nuc = fc.getBackgroundLevels(nuclei_ds)[1]
    bkg_cyto = fc.getBackgroundLevels(cyto_ds)[1]

    dataQueue = mp.Queue()
    save_thread = mp.Process(target=saveProcess, args=[dataQueue])
    save_thread.start()

    # create reference to full res data
    nuclei_hires = f['/t00000/s00/0/cells']
    cyto_hires = f['/t00000/s01/0/cells']

    # block size for Image data
    tileSize = 256

    # settings for RGB conversion
    settings_dict = fc.getColorSettings()
    nuclei_RGBsettings = settings_dict['nuclei']
    cyto_RGBsettings = settings_dict['cyto']
    print(nuclei_RGBsettings)
    print(cyto_RGBsettings)

    for k in range(start_k, stop_k, skip_k):
        if k == stop_k:
            break
        else:

            t_start = time.time()

            print('on section: ', k)

            # get image data from both channels in blocks that are
            # multiples of tileSize
            # subtract background and reset values > 0 and < 2**16
            print('Reading Data')
            t_nuc = time.time()
            nuclei = nuclei_hires[0:tileSize*M_nuc.shape[0], k,
                                  0:tileSize*M_nuc.shape[2]].astype(numpy.uint16)
            nuclei = nuclei.astype(float)
            nuclei -= 0.5*bkg_nuc
            nuclei = numpy.clip(nuclei, 0, 65535)
            print('read time nuclei', time.time()-t_nuc)

            t_cyt = time.time()
            cyto = cyto_hires[0:tileSize*M_cyto.shape[0], k,
                              0:tileSize*M_cyto.shape[2]].astype(numpy.uint16)
            cyto = cyto.astype(float)
            cyto -= 3*bkg_cyto
            cyto = numpy.clip(cyto, 0, 65535)
            print('read time cyto', time.time() - t_cyt)

            # sharpen images
            print('sharpening')
            nuclei = fc.sharpenImage(nuclei, alpha=alpha)
            cyto = fc.sharpenImage(cyto, alpha=alpha)

            # interpolate downsampled data to full res to use as leveling map
            C_nuc = fc.interpolateDS(M_nuc, k, beta=nuc_norm_constant)
            C_cyto = fc.interpolateDS(M_cyto, k, beta=cyto_norm_constant)

            print('False Coloring')

            # Execute false coloring method
            RGB_image = fc.rapidFalseColor(nuclei, cyto,
                                           nuclei_RGBsettings,
                                           cyto_RGBsettings,
                                           nuc_normfactor=C_nuc,
                                           cyto_normfactor=C_cyto,
                                           run_FlatField_nuc=True,
                                           run_FlatField_cyto=True)

            # append data to queue
            save_file = '{:0>6d}'.format(k) + args.format
            message = [filepath, save_dir, save_file, RGB_image, None]

            dataQueue.put(message)

            nuclei = None
            cyto = None
            print('runtime:', time.time() - t_start)

    # stop data queue
    stop_message = [None, None, None, None, 'stop']
    dataQueue.put(stop_message)
    save_thread.join()
    f.close()


if __name__ == '__main__':
    t_overall = time.time()
    print('Starting False Color Script')
    main()
    print('total runtime: %s minutes' % ((time.time()-t_overall)/60))
