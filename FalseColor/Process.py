"""
#===============================================================================
# 
#  License: GPL
#
#
#  Copyright (c) 2019 Rob Serafin, Liu Lab, 
#  The University of Washington Department of Mechanical Engineering  

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
import scipy.ndimage as nd
import skimage.filters as filt
import skimage.exposure as ex
import skimage.util as util
import cv2
import numpy
import h5py as hp
from numba import cuda, njit
import json


@njit
def sortImage(image, mask_val = 255):
    """
    Method for sorting image pixel values excluding a high value threshold. Used by getRGBStats to 
    get a 1D array for pixel values less than saturated value.

    Parameters
    ----------

    image : 2D numpy array
        Image array 

    mask_val : int
        High threshold over which pixel values will be ignored


    Returns
    -------
    pixel_set : 1D numpy array
        Pixel values contained in image which are under the mask value.

    """
    pixel_set = []

    #Ravel image and sort through item by item
    image_list = image.ravel()
    for item in image_list:
        if item < mask_val:
            #append to pixel set if value is under threshold
            pixel_set.append(item)
    return numpy.asarray(pixel_set)

def getRGBStats(image, mask_val = 255):
    """
    Method which returns dictionary containing useful image statistics from the input RGB image. 
    Will be called from a save metadata method which will store data in a json.


    Parameters
    ----------

    image : 3D numpy array
        RGB image array 

    mask_val : int
        High threshold over which pixel values will be ignored

    Returns
    -------

    image_stats : dict
        Dictionary with 'R', 'G', 'B' keys which each have a dict as follows:

            'data' : data called from sortImage method, below mask value

            'median' : median of sorted image data

            '90th' : 90th percentile of sorted image data

            '10th' : 10th percentile of sorted image data

    """
    R = {'data' : sortImage(image[:,:,0], mask_val = mask_val)}
    G = {'data' : sortImage(image[:,:,1], mask_val = mask_val)}
    B = {'data' : sortImage(image[:,:,1], mask_val = mask_val)}
    
    R['median'] = numpy.median(R['data'])
    G['median'] = numpy.median(G['data'])
    B['median'] = numpy.median(B['data'])

    R['90th'] = numpy.percentile(R['data'], 90)
    R['10th'] = numpy.percentile(R['data'], 10)

    G['90th'] = numpy.percentile(G['data'], 90)
    G['10th'] = numpy.percentile(G['data'], 10)

    B['90th'] = numpy.percentile(B['data'], 90)
    B['10th'] = numpy.percentile(B['data'], 10)

    image_stats = {'R' : R, 'G' : G, 'B' : B}

    return image_stats

def saveRGBStats(image_stats,folder,filename):
    """
    Parameters
    ----------

    image_stats : dict
        dictionary of the same form as the one returned from getRGBStats

    folder : str or path
        string or pathlike object corresponding to the storage directory where the npz file 
        will be saved

    filename : str
        name of file to be saved

    Returns
    -------

    """

    savepath = os.path.join(folder,filename)

    numpy.savez(savepath, R = image_stats['R'], G = image_stats['G'], B = image_stats['B'])
