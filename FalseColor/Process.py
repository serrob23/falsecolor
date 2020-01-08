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
import FalseColor.Color as fc
import cv2
import numpy
import h5py as hp
from numba import cuda, njit
import json
import matplotlib.pyplot as plt


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

def getHSVstats(nuclei, cyto, 
                    hue_mask_value = 0, 
                    sat_mask_value = 0, 
                    val_mask_value = 0):
    """
    Method which returns dictionary containing useful image statistics from the input RGB image. 
    Will be called from a save metadata method which will store data in a json.


    Parameters
    ----------

    nuclei : 3D numpy array
        Image array to analyze which has been converted from RGB to HSV space, representing the
    hematoxylin, or equivalent, segmented from a histology or virually stained image. 

    cyto : 3D numpy array
        Image array to analyze which has been converted from RGB to HSV space, representing the
    eosin, or equivalent, segmented from a histology or virually stained image.

    mask_val : int
        High threshold over which pixel values will be ignored

    Returns
    -------

    image_stats : dict
        Dictionary with 'Hue', 'Sat', 'Val' keys which each have a dict as follows:

            'median' : median of sorted image data

            '90th' : 90th percentile of sorted image data

            '10th' : 10th percentile of sorted image data

            'std' : standard deviation of sorted image data
    """

    H_nuc = sortImage(nuclei[:,:,0], mask_val = hue_mask_value, greater_mode = True)
    S_nuc = sortImage(nuclei[:,:,1], mask_val = sat_mask_value, greater_mode = True)
    V_nuc = sortImage(nuclei[:,:,2], mask_val = val_mask_value, greater_mode = True)
    
    H_cyto = sortImage(cyto[:,:,0], mask_val = hue_mask_value, greater_mode = True)
    S_cyto = sortImage(cyto[:,:,1], mask_val = sat_mask_value, greater_mode = True)
    V_cyto = sortImage(cyto[:,:,2], mask_val = val_mask_value, greater_mode = True)

    image_stats = {'nuclei' : {'Hue' : {}, 'Sat' : {}, 'Val' : {}},
                    'cyto' : {'Hue' : {}, 'Sat' : {}, 'Val' : {}}}

    image_stats['nuclei']['Hue']['median'] = numpy.median(H_nuc)
    image_stats['nuclei']['Hue']['10'] = numpy.percentile(H_nuc, 10)
    image_stats['nuclei']['Hue']['90'] = numpy.percentile(H_nuc, 90)
    image_stats['nuclei']['Hue']['std'] = numpy.std(H_nuc)

    image_stats['nuclei']['Sat']['median'] = numpy.median(S_nuc)
    image_stats['nuclei']['Sat']['10'] = numpy.percentile(S_nuc, 10)
    image_stats['nuclei']['Sat']['90'] = numpy.percentile(S_nuc, 90)
    image_stats['nuclei']['Sat']['std'] = numpy.std(S_nuc)

    image_stats['nuclei']['Val']['median'] = numpy.median(V_nuc)
    image_stats['nuclei']['Val']['10'] = numpy.percentile(V_nuc, 10)
    image_stats['nuclei']['Val']['90'] = numpy.percentile(V_nuc, 90)
    image_stats['nuclei']['Val']['std'] = numpy.std(V_nuc)

    image_stats['cyto']['Hue']['median'] = numpy.median(H_cyto)
    image_stats['cyto']['Hue']['10'] = numpy.percentile(H_cyto, 10)
    image_stats['cyto']['Hue']['90'] = numpy.percentile(H_cyto, 90)
    image_stats['cyto']['Hue']['std'] = numpy.std(H_cyto)

    image_stats['cyto']['Sat']['median'] = numpy.median(S_cyto)
    image_stats['cyto']['Sat']['10'] = numpy.percentile(S_cyto, 10)
    image_stats['cyto']['Sat']['90'] = numpy.percentile(S_cyto, 90)
    image_stats['cyto']['Sat']['std'] = numpy.std(S_cyto)

    image_stats['cyto']['Val']['median'] = numpy.median(V_cyto)
    image_stats['cyto']['Val']['10'] = numpy.percentile(V_cyto, 10)
    image_stats['cyto']['Val']['90'] = numpy.percentile(V_cyto, 90)
    image_stats['cyto']['Val']['std'] = numpy.std(V_cyto)

    return image_stats

def saveImageStats(image_stats,folder,filename):
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

    with open(savepath, 'w') as f:
        json.dump(image_stats,f)
    f.close()


def ViewImage(Image, title=None, do_hist = False, figsize = (6,4), 
                    range_min=0, range_max=None, cmap='viridis', do_ticks = False):

    """
    Method for viewing images easily.

    Parameters 
    ----------

    Image : 2D numpy array
        Image to view.

    title : None or str
        default to None, if not will be used as plot title.

    do_hist : bool
        default to False, if True will plot histogram of image in an adjacent subplot.

    figsize : tuple
        defaults to (6,4), tuple of ints which defines size of plotted image. 

    range_min : int
        defaults to 0, minimum range for histogram. 

    range_max : None or int
        defaults to None, if int maximum range for histogram.

    cmap : str
        defaults to 'viridis', colormap for plotting grayscale images. 

    do_ticks : bool
        defaults to False, plot x/y ticks and ticklabels on plot.

    Returns
    -------
    f,ax : matplotlib figure, matplotlib axis
        figure and axis

    """
    
    if do_hist:
        
        f,ax = plt.subplots(ncols = 2,figsize = figsize)
        
        #plot image
        ax[0].imshow(Image, cmap = cmap, vmin = Image.min(), vmax = Image.max())
        ax[0].set_title('Image')
        
        #setup histogram
        if range_max is None:
            range_max = Image.max()
        ax[1].hist(Image[Image != 0].ravel(),256,[range_min, range_max])

        ax[1].set_title('Histogram')
        ax[1].set_xlabel('Intenstity, A.U')

        asp = numpy.diff(ax[1].get_xlim())[0] / numpy.diff(ax[1].get_ylim())[0]

        ax[1].set_aspect(asp)
        plt.subplots_adjust(wspace = 0.5)

        #set title
        if title is not None:
            f.suptitle(title)
            
    else:

        #plot image
        f,ax = plt.subplots(figsize = figsize)
        ax.imshow(Image, cmap = cmap)
            
        #set title
        if title is not None:
            ax.set_title(title)
        
            
    if do_ticks == False:

        #adjust tick parameters
        plt.tick_params(axis = 'both' , which = 'both', top = False,
                        bottom = False, left = False, right = False,
                        labelbottom = False, labelleft = False)
    return f,ax
    
