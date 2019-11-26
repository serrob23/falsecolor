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
from skimage import color
import cv2
import numpy
import h5py as hp
from numba import cuda, njit
import json


@njit
def sortImage(image, mask_val = 255, greater_mode = False):
    """
    Method for sorting image pixel values excluding a high value threshold. Used by getRGBStats to 
    get a 1D array for pixel values less than saturated value.

    Parameters
    ----------

    image : 2D numpy array
        Image array 

    mask_val : int
        High threshold over which pixel values will be ignored

    greater_mode: bool
        default = False, whether to filter out values less than mask_val


    Returns
    -------
    pixel_set : 1D numpy array
        Pixel values contained in image which are under the mask value.
    """
    pixel_set = []

    #Ravel image and sort through item by item
    image_list = image.ravel()
    for item in image_list:
        if not greater_mode:
            if item < mask_val:
                #append to pixel set if value is under threshold
                pixel_set.append(item)
        elif greater_mode: 
            if item > mask_val:
                pixel_set.append(item)
        else:
            if item != mask_val:
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
    B = {'data' : sortImage(image[:,:,2], mask_val = mask_val)}
    
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

def getHSstats(nuclei, cyto, hue_mask_value = 0, sat_mask_value = 0,
                        color_change = True):

    if color_change:
        nuclei = color.rgb2hsv(nuclei)
        cyto = color.rgb2hsv(cyto)

    H_nuc = sortImage(nuclei[:,:,0], mask_val = hue_mask_value, greater_mode = True)
    S_nuc = sortImage(nuclei[:,:,1], mask_val = sat_mask_value, greater_mode = True)
    
    H_cyto = sortImage(cyto[:,:,0], mask_val = hue_mask_value, greater_mode = True)
    S_cyto = sortImage(cyto[:,:,1], mask_val = sat_mask_value, greater_mode = True)

    image_stats = {'nuclei' : {'Hue' : H_nuc, 'Saturation' : S_nuc},
                    'cyto' : {'Hue' : H_cyto, 'Saturation' : S_cyto}}

    image_stats['nuclei']['H_median'] = numpy.median(H_nuc)
    image_stats['nuclei']['H_10th'] = numpy.percentile(H_nuc, 10)
    image_stats['nuclei']['H_90th'] = numpy.percentile(H_nuc, 90)

    image_stats['nuclei']['S_median'] = numpy.median(S_nuc)
    image_stats['nuclei']['S_10th'] = numpy.percentile(S_nuc, 10)
    image_stats['nuclei']['S_90th'] = numpy.percentile(S_nuc, 90)

    image_stats['cyto']['H_median'] = numpy.median(H_cyto)
    image_stats['cyto']['H_10th'] = numpy.percentile(H_cyto, 10)
    image_stats['cyto']['H_90th'] = numpy.percentile(H_cyto, 90)

    image_stats['cyto']['S_median'] = numpy.median(S_cyto)
    image_stats['cyto']['S_10th'] = numpy.percentile(S_cyto, 10)
    image_stats['cyto']['S_90th'] = numpy.percentile(S_cyto, 90)

    return image_Stats

def saveImageStats(image_stats,folder,filename, RGB = True):
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

    if RGB:
        numpy.savez(savepath, R = image_stats['R'], G = image_stats['G'], B = image_stats['B'])

    else:
        numpy.savez(savepath, nuclei = image_stats['nuclei'], cyto = image_stats['cyto'])
