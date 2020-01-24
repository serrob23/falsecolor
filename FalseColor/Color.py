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
import scipy.ndimage as nd
import skimage.filters as filt
import skimage.exposure as ex
import skimage.util as util
import skimage.morphology as morph
from skimage.color import rgb2hed, rgb2hsv, hsv2rgb
import cv2
import numpy
from numba import cuda, njit
import math
from astropy.convolution import Gaussian2DKernel


@cuda.jit #direct GPU compiling
def rapid_getRGBframe(nuclei, cyto, output, nuc_settings,
                        cyto_settings, k_nuclei, k_cyto):
    #TODO: implement array base normalization
    """
    GPU based exponential false coloring operation. Used by rapidFalseColor()

    Parameters
    ----------
    nuclei : 2D numpy array 
        dtype = float
        Nuclear channel image, already pre processed

    cyto : 2d numpy array
        dtype = float
        Cytoplasm channel image, already pre processed.
        
    nuc_settings : float
        RGB constant for nuclear channel
    
    cyto_settings : float
        RGB constant for cyto channel

    k_nuclei : float
        Additional multiplicative constant for nuclear channel. Eventually will get removed once 
        flat fielding is in place for all pseudo coloring methods.

    k_cyto: float
        Additional multiplicative constant for cytoplasmic channel. Eventually will get removed once 
        flat fielding is in place for all pseudo coloring methods.        
    """
    row,col = cuda.grid(2)

    #iterate through image and assign pixel values
    if row < output.shape[0] and col < output.shape[1]:
        tmp = nuclei[row,col]*nuc_settings*k_nuclei + cyto[row,col]*cyto_settings*k_cyto
        output[row,col] = 255*math.exp(-1*tmp)


@cuda.jit
def rapidFieldDivision(image,flat_field,output):
    """
    Used for rapidFalseColoring() when flat field has been calculated

    Parameters
    ----------

    image : numpy array written to GPU

    flat_field : numpy array written to GPU

    output : numpy array written to GPU
        result from computation

    """
    row,col = cuda.grid(2)

    if row < output.shape[0] and col < output.shape[1]:
        tmp = image[row,col]/flat_field[row,col]
        output[row,col] = tmp

def rapidFalseColor(nuclei, cyto, nuc_settings, cyto_settings,
                   TPB = (32,32), nuc_normfactor = 8500, cyto_normfactor = 3000,
                   run_FlatField_nuc = False, run_FlatField_cyto = False, nuc_bg_threshold = 50,
                   cyto_bg_threshold = 50):
    """
    Parameters
    ----------

    nuclei : numpy array
        Nuclear channel image.
        
    cyto : numpy array
        Cytoplasm channel image.
        
    nuc_settings : list
        Settings of RGB constants for nuclear channel. Should be in order R, G, B.
    
    cyto_settings : list
        Settings of RGB constants for cytoplasm channel. Should be in order R, G, B.

    nuc_normfactor : int or array
        Defaults to empirically determined constant to reduce saturation. Otherwise it should be a 
        numpy array representing the true flat field image.

    cyto_normfactor : int or array
        Defaults to empirically determined constant to reduce saturation. Otherwise it should be a 
        numpy array representing the true flat field image.
        
    TPB : tuple (int,int)
        THREADS PER BLOCK: (x_threads,y_threads) used for GPU threads.

    run_FlatField : bool
        defaults to False, boolean to apply flatfield

    nuc_bg_threshold = int
        defaults to 50, threshold level for calculating nuclear background

    cyto_bg_threshold = int
        defaults to 50, threshold level for calculating cytoplasmic background



    Returns
    -------
    RGB_image : 3D numpy array
        Combined false colored image in the standard RGB format [X, Y, C]

    """

    #ensure float dtype
    nuclei = nuclei.astype(float)
    cyto = cyto.astype(float)

    #set mulciplicative constants, changes on flat fielding vs background subtraction
    k_nuclei = 1.0
    k_cyto = 1.0

    #create blockgrid for gpu
    blockspergrid_x = int(math.ceil(nuclei.shape[0] / TPB[0]))
    blockspergrid_y = int(math.ceil(nuclei.shape[1] / TPB[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    #allocate memory for background subtraction
    nuclei = numpy.ascontiguousarray(nuclei)
    pre_nuc_output = cuda.to_device(numpy.zeros(nuclei.shape))
    nuc_global_mem = cuda.to_device(nuclei)

    #run background subtraction or normalization for nuclei

    #use flat fielding
    if run_FlatField_nuc:
        nuc_normfactor = numpy.ascontiguousarray(nuc_normfactor)
        nuc_norm_mem = cuda.to_device(nuc_normfactor)
        rapidFieldDivision[blockspergrid,TPB](nuc_global_mem,nuc_norm_mem,pre_nuc_output)

    #otherwise use standard background subtraction
    else:
        k_nuclei = 0.08
        nuc_background = getBackgroundLevels(nuclei, threshold = nuc_bg_threshold)[1]
        rapid_preProcess[blockspergrid,TPB](nuc_global_mem,nuc_background,
                                                nuc_normfactor,pre_nuc_output)
    
    #allocate memory for background subtraction
    cyto = numpy.ascontiguousarray(cyto)
    pre_cyto_output = cuda.to_device(numpy.zeros(cyto.shape))
    cyto_global_mem = cuda.to_device(cyto)

    #run background subtraction or normalization for cyto

    #use flat fielding
    if run_FlatField_cyto:
        cyto_normfactor = numpy.ascontiguousarray(cyto_normfactor)
        cyto_norm_mem = cuda.to_device(cyto_normfactor)
        rapidFieldDivision[blockspergrid,TPB](cyto_global_mem,cyto_norm_mem,pre_cyto_output)

    # otherwise use standard background subtraction
    else:
        k_cyto = 0.012
        cyto_background = getBackgroundLevels(cyto, threshold = cyto_bg_threshold)[1]
        rapid_preProcess[blockspergrid,TPB](cyto_global_mem,cyto_background,
                                                cyto_normfactor,pre_cyto_output)
    
    #create output array to iterate through
    RGB_image = numpy.zeros((3,nuclei.shape[0],nuclei.shape[1]), dtype = numpy.int8) 

    #iterate through output array and assign values based on RGB settings
    for i,z in enumerate(RGB_image): #TODO: speed this up on GPU

        #allocate memory on GPU with background subtracted images and final output
        output_global = cuda.to_device(numpy.zeros(z.shape)) 
        nuclei_global = cuda.to_device(pre_nuc_output)
        cyto_global = cuda.to_device(pre_cyto_output)

        #get 8bit frame
        rapid_getRGBframe[blockspergrid,TPB](nuclei_global,cyto_global,output_global,
                                                nuc_settings[i],cyto_settings[i],
                                                k_nuclei, k_cyto)
        
        RGB_image[i] = output_global.copy_to_host()

    #reorder array to dimmensional form [X,Y,C]
    RGB_image = numpy.moveaxis(RGB_image,0,-1)
    return RGB_image.astype(numpy.uint8)


@cuda.jit #direct GPU compiling
def rapid_preProcess(image,background,norm_factor,output):
    """
    Background subtraction optimized for GPU, used by rapidFalseColor.

    Parameters
    ----------

    image : 2d numpy array, dtype = int16
        Image for background subtraction.

    background : int
        Constant for subtraction.

    norm_factor : int
        Empirically determaned constant for normalization after subtraction. Helps prevent 
    saturation.

    output : 2d numpy array
        Numpy array of zeros for GPU to assign values to.

    Returns
    -------
    This method requires an output array as an argument, the results of the compuation are stored 
    there.
    """

    #create iterator for gpu  
    row,col = cuda.grid(2)

    #cycle through image shape and assign values
    if row < output.shape[0] and col < output.shape[1]:

        #subtract background and raise to factor
        tmp = image[row,col] - background

        #remove negative values
        if tmp < 0:
            output[row,col] = 0

        #normalize to 8bit range
        else:
            tmp = (tmp**0.85)*(65535/norm_factor)*(255/65535)
            output[row,col] = tmp


@cuda.jit
def Convolve2d(image,kernel,output):
    """
    GPU based 2d convolution method.

    Parameters
    ----------

    image : 2D numpy array
        Image for processing, written to GPU.

    kernel : 2D numpy array
        kernel to convolve image with, written to GPU

    output : 2D numpy array
        Output array, storing result of convolution, written to GPU.

    Returns
    -------
    This method requires an output array as an argument, the results of the compuation are stored 
    there.
    """

    #create iterator
    row,col = cuda.grid(2)
    
    image_rows,image_cols = image.shape
    
    delta_r = kernel.shape[0]//2
    delta_c = kernel.shape[1]//2
    
    #ignore rows/cols outside image
    if (row >= image_rows) or (col >= image_cols):
        return
    
    tmp = 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            #result should be sum of kernel*image as kernel is varied
            row_i = row - i + delta_r
            col_j = col - j + delta_c
            if (row_i>=0) and (row_i < image_rows):
                if (col_j>=0) and (col_j < image_cols):
                    tmp += kernel[i,j]*image[row_i,col_j]
                    
    output[row,col] = tmp 


def sharpenImage(input_image,alpha = 0.5):
    """
    Image sharpening algorithm to amplify edges.

    Parameters
    ----------

    input_image : 2D numpy array
        Image to run sharpening algorithm on

    alpha : float or int
        Multiplicative constant for final result.

    Returns
    --------

    final_image : 2D numpy array
        The sum of the input image and the resulting convolutions
    """
    #create kernels to amplify edges
    hkernel = numpy.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    vkernel = numpy.array([[1,0,-1],[1,0,-1],[1,0,-1]])

    #set grid/threads for GPU
    blocks = (32,32)
    grid = (input_image.shape[0]//blocks[0] + 1, input_image.shape[1]//blocks[1] + 1)

    #run convolution
    input_image = numpy.ascontiguousarray(input_image)
    voutput = numpy.zeros(input_image.shape,dtype=numpy.float64)
    houtput = numpy.zeros(input_image.shape,dtype=numpy.float64)
    Convolve2d[grid,blocks](input_image,vkernel,voutput)
    Convolve2d[grid,blocks](input_image,hkernel,houtput)

    #calculate final result
    final_image = input_image + 0.5*numpy.sqrt(voutput**2 + houtput**2)
    
    return final_image


def gaussianBlur(image, sigma = 5):
    """
    Cuda accelerated gaussian blurring using Convolve2D.

    Parameters
    ----------

    image : numpy array
        Image for blurring

    sigma : int
        Standard deviation of 2D gaussian kernel


    Returns
    -------

    blurred_image : numpy array
        image convolved with gaussian kernel

    """

    #ensure image dtype is float
    if image.dtype != float:
        image = image.astype(float)

    #create 2D gaussian kernel and convert to numpy array
    gaussian_kernel = Gaussian2DKernel(sigma)
    gaussian_kernel = numpy.asarray(gaussian_kernel, dtype = float)

    #create block/grid for cuda.jit
    blocks = (32,32)
    grid = (image.shape[0]//blocks[0] + 1, image.shape[1]//blocks[1] + 1)

    #create output array
    blurred_image = numpy.zeros(image.shape)

    #run convolution
    image = numpy.ascontiguousarray(image)
    Convolve2d[grid,blocks](image, gaussian_kernel, blurred_image)

    return blurred_image

def getDefaultRGBSettings(use_default = True):

    """Returns empirically determined constants for nuclear/cyto channels

    Parameters
    ----------

    use_default : bool
        Defaults to True. Which RGB settings to use, when False will use RGB settings which are
        empirically derived from histology color analysis. Note: these settings currently only 
        optimized for flat field method in rapidFalseColor. 


    Returns
    -------
    settings_dict : dict
        Dictionary with keys 'nuclei', 'cyto' which correspond to lists containing empirically 
        derived RGB constants for false coloring.


    """
    if use_default:
        k_cyto = 1.0
        k_nuclei = 0.85
        nuclei_RGBsettings = [0.25*k_nuclei, 0.37*k_nuclei, 0.1*k_nuclei]
        cyto_RGB_settings = [0.05*k_cyto, 1.0*k_cyto, 0.54*k_cyto]

        settings_dict = {'nuclei':nuclei_RGBsettings,'cyto':cyto_RGB_settings}
    else:
        k_cyto = 1.0
        nuclei_RGBsettings = [0.17, 0.27, 0.1]
        cyto_RGB_settings = [0.05*k_cyto, 1.0*k_cyto, 0.54*k_cyto]
        settings_dict = {'nuclei':nuclei_RGBsettings,'cyto':cyto_RGB_settings}
    return settings_dict


def applyCLAHE(image, clahe = None, tileGridSize = (8,8), 
                                    input_dtype = numpy.uint16,
                                    clipLimit = 0.048):
    """
    Applies Contrast Limited Adaptive Histogram Equalization algorithm from OpenCV. 

    Parameters 
    ----------

    image : 2D numpy array
        Image for histogram equalization

    clahe : None or cv2.CLAHE object
        CV2 object to use for equalization

    tileGridSize : tuple
        Tuple of ints representing the windowsize for CLAHE, default is (32,32)

    input_dtype : numpy dtype
        Dtype to use for CLAHE object, defaults to numpy.uint16, cv2 CLAHE is compatible with either
        numpy.uint8 or numpy.uint16

    Returns
    -------

    equalized_image : 2D numpy array
        Image with equalized historgram.

    """

    if clahe is None:
        #create clahe object
        clahe = cv2.createCLAHE(tileGridSize = tileGridSize, clipLimit = clipLimit)

    #ensure image is of uint dtype
    image = image.astype(input_dtype)

    #apply CLAHE
    equalized_image = clahe.apply(image)

    #Renormalize to original image levels
    final_image = (image.max())*(equalized_image/equalized_image.max())

    return final_image


def falseColor(nuclei, cyto, output_dtype=numpy.uint8, 
                    nuc_bg_threshold = 50, cyto_bg_threshold = 50, nuc_normfactor = None, cyto_normfactor=None):
    """
    Two channel virtual H&E coloring using Beer's law method based on:
    Giacomelli et al., PLOS one 2016 doi:10.1371/journal.pone.0159337.


    Parameters
    ----------
    nuclei : 2D numpy array
       Image of nuclear stain (hematoxylin equivalent) for the Virtual H&E transformation.

    cyto : 2D numpy array
        Image of cytoplasm stain (eosin equivalent) for the Virtual H&E transformation

    channelIDs = list
        keys to grab settings from beta_dict
        defaults: s00 : nuclei
                  s01 : cyto

    output_dtype : numpy.uint8
        output datatype for final RGB image

    nuc_bg_threshold = int
        defaults to 50, threshold level for calculating nuclear background

    cyto_bg_threshold = int
        defaults to 50, threshold level for calculating cytoplasmic background


    Returns
    -------
    RGB_image : numpy array
        Combined false colored image in the standard RGB format [X, Y, C]

    """
    beta_dict = {
                #constants for nuclear channel
                'K_nuclei' : 0.08,

                #constants for cytoplasmic channel
                'K_cyto' : 0.0120}

    #returns dictionary with settings for each channel
    #keys are: nuclei, cyto
    #entries are lists in order of RGB constants
    settings = getDefaultRGBSettings()

    constants_nuclei = settings['nuclei']
    k_nuclei = beta_dict['K_nuclei']

    constants_cyto = settings['cyto']
    k_cytoplasm= beta_dict['K_cyto']
    
    #execute background subtraction
    nuclei = nuclei.astype(float)
    nuc_threshold = getBackgroundLevels(nuclei, nuc_bg_threshold)[1]
    nuclei = preProcess(nuclei, threshold = nuc_threshold, normfactor = nuc_normfactor)

    cyto = cyto.astype(float)
    cyto_threshold = getBackgroundLevels(cyto, cyto_bg_threshold)[1]
    cyto = preProcess(cyto, threshold = cyto_threshold, normfactor = cyto_normfactor)

    RGB_image = numpy.zeros((3,nuclei.shape[0],nuclei.shape[1]))

    #iterate throough RGB constants and execute image multiplication
    for i in range(len(RGB_image)):
        RGB_image[i] = 255*numpy.multiply(numpy.exp(-constants_cyto[i]*k_cytoplasm*cyto),
                                        numpy.exp(-constants_nuclei[i]*k_nuclei*nuclei))

    #reshape to [X,Y,C]
    RGB_image = numpy.moveaxis(RGB_image,0,-1)

    #rescale to 8bit range
    return RGB_image.astype(output_dtype)


def preProcess(image, threshold = 50, normfactor = None):
    """
    Method used for background subtracting data with a fixed value

    Parameters
    ----------

    image : 2D numpy array
        image for processing

    threshold : int
        background level to subtract

    Returns
    -------

    processed_image : 2D numpy array
        Background subtracted image. 
    """

    #background subtraction
    image -= threshold

    #no negative values
    image[image < 0] = 0

    #calculate normalization factor
    image = numpy.power(image,0.85)
    if normfactor is None:
        normfactor = numpy.mean(image[image>threshold])*8

    #convert into 8bit range
    processed_image = image*(65535/normfactor)*(255/65535)

    return processed_image


def getBackgroundLevels(image, threshold = 50):
    """
    Calculate foreground and background values based on image statistics, background is currently
    set to be 20% of foreground

    Parameters
    ----------

    image : 2D numpy array

    threshold : int
        threshold above which is counted as foreground

    Returns
    -------

    hi_val : int
        Foreground values

    background : int
        Background value
    """

    image_DS = numpy.sort(image,axis=None)

    foreground_vals = image_DS[numpy.where(image_DS > threshold)]

    hi_val = foreground_vals[int(numpy.round(len(foreground_vals)*0.95))]

    background = hi_val/5

    return hi_val,background


def getFlatField(image,tileSize=256,blockSize = 16, bg_threshold = 50):

    """
    Returns downsampled flat field of image data and calculated background levels

    Parameters
    ----------

    image : 2D or 3D numpy array

    tileSize : int

    blockSize : int

    Returns
    -------

    flat_field : 2D numpy array
        Calculated flat field for input image

    background : float
        Background level for input image
    """

    midrange,background = getBackgroundLevels(image, threshold = bg_threshold)
    
    rows_max = int(numpy.ceil(image.shape[0]/blockSize)*blockSize)
    cols_max = int(numpy.ceil(image.shape[2]/blockSize)*blockSize)
    stacks_max = int(numpy.ceil(image.shape[1]/blockSize)*blockSize)


    rows = numpy.arange(0, rows_max+int(tileSize/blockSize), int(tileSize/blockSize))
    cols = numpy.arange(0, cols_max+int(tileSize/blockSize), int(tileSize/blockSize))
    stacks = numpy.arange(0, stacks_max+int(tileSize/blockSize), int(tileSize/blockSize))
    
    flat_field = numpy.zeros((len(rows)-1, len(stacks)-1, len(cols)-1), dtype = float)
    
    for i in range(1,len(rows)):
        for j in range(1,len(stacks)):
            for k in range(1,len(cols)):

                ROI_0 = image[rows[i-1]:rows[i], stacks[j-1]:stacks[j], cols[k-1]:cols[k]]
                
                fkg_ind = numpy.where(ROI_0 > background)
                if fkg_ind[0].size==0:
                    Mtemp = midrange
                else:
                    Mtemp = numpy.median(ROI_0[fkg_ind])
                flat_field[i-1, j-1, k-1] = Mtemp + flat_field[i-1, j-1, k-1]
    return flat_field, background/5


def interpolateDS(M_nuc, M_cyt, k, tileSize = 256):
    """
    Method for resizing downsampled data to be the same size as full res data. Used for 
    interpolating flat field images.

    Parameters
    ----------

    M_nuc : 2D numpy array
        Downsampled data from BigStitcher 

    M_cyt : 2D numpy array
        Downsampled data from BigStitcher

    k : int
        Index for image location in full res HDF5 file

    tileSize : int
        Default = 256, block size for interpolation

    Returns
    -------

    C_nuc : 2D numpy array
        Rescaled downsampled data

    C_cyt : 2D numpy array
        Rescaled downsampled data

    """

    x0 = numpy.ceil(k/tileSize)
    x1 = numpy.ceil(k/tileSize)
    x = k/tileSize

    #get background block
    if k < int(M_nuc.shape[1]*tileSize-tileSize):
        if k < int(tileSize/2):
            C_nuc = M_nuc[:,0,:]
            C_cyt = M_cyt[:,0,:]

        elif x0==x1:
            C_nuc = M_nuc[:,int(x1),:]
            C_cyt = M_cyt[:,int(x1),:]
        else:
            nuc_norm0 = M_nuc[:,int(x0),:]
            nuc_norm1 = M_nuc[:,int(x1),:]

            cyto_norm0 = M_cyt[:,int(x0),:]
            cyto_norm1 = M_cyt[:,int(x1),:]

            C_nuc = nuc_norm0 + (x-x0)*(nuc_norm1 - nuc_norm0)/(x1-x0)
            C_cyt = cyto_norm0 + (x-x0)*(cyto_norm1 - cyto_norm0)/(x1-x0)
    else:
        C_nuc = M_nuc[:,M_nuc.shape[1]-1, :]
        C_cyt = M_cyt[:,M_cyt.shape[1]-1, :]

    C_nuc = nd.interpolation.zoom(C_nuc, tileSize, order = 1, mode = 'nearest')

    C_cyt = nd.interpolation.zoom(C_cyt, tileSize, order = 1, mode = 'nearest')

    return C_nuc, C_cyt


def deconvolveColors(image):
    """
    Separates H&E channels from an RGB image using skimage.color.rgb2hed method

    Parameters
    ----------

    image : 3D numpy array
        RGB image in the format [X, Y, C] where the hematoxylin and eosin channels 
        are to be separted.

    Returns
    -------

    hematoxylin : 2D numpy array
        nuclear channel deconvolved from RGB image


    eosin : 2D numpy array
        cytoplasm channel deconvolved from RGB image

    """

    separated_image = rgb2hed(image)

    hematoxylin = separated_image[:,:,0]

    eosin = separated_image[:,:,1]

    return hematoxylin, eosin


def segmentNuclei(image, return3D = True, opening = True, 
                         radius = 3, min_size = 64,
                         return_cyto = False):
    """
    
    Grabs binary mask of nuclei from H&E image using color deconvolution. 

    Parameters
    ----------

    image : 3D numpy array 
        H&E stained RGB image in the form [X, Y, C]

    return3D : bool, 
        Defaults to False, return 3D version of mask

    return_cyto : bool
        Defaults to False, will return a binary mask for cytoplasm from color deconvolved RGB image

    Returns
    -------

    binary_mask : 2D or 3D numpy array
        Binary mask of segmented nuclei

    """

    #separate channels
    nuclei, cyto = deconvolveColors(image)

    #median filter nuclei for optimized otsu threshold
    median_filtered_nuclei = filt.median(nuclei)

    #calculate threshold and create initial binary mask
    threshold = filt.threshold_otsu(median_filtered_nuclei)
    binarized_nuclei = (median_filtered_nuclei > threshold).astype(int)

    #remove small objects
    labeled_mask = morph.label(binarized_nuclei)
    shape_filtered_mask = morph.remove_small_objects(labeled_mask, min_size = min_size)

    #binary opening to separate objects
    if opening:
        shape_filtered_mask = morph.binary_opening(shape_filtered_mask, morph.disk(radius))

    #remove labels from nuclear mask
    binary_mask = (shape_filtered_mask > 0)

    #create a binary mask for cytoplasm
    if return_cyto:

        #median filter cyto for optimized otsu threshold
        median_filtered_cyto = filt.median(cyto)

        #calculate threshold and create initial binary mask
        cyto_threshold = filt.threshold_otsu(median_filtered_cyto)

        #create binary mask
        binary_cyto = (median_filtered_cyto > cyto_threshold).astype(int)

        #ensure that nuclei are segmented out of cyto mask
        binary_cyto = binary_cyto*(binary_mask<1)

        # binary closing to remove pepper noise
        if opening:
            binary_cyto = morph.binary_closing(binary_cyto, morph.disk(radius))


        #create 3D array and rearrange shape to match an RGB image
        if return3D:
            binary_cyto = numpy.moveaxis(numpy.asarray([binary_cyto, binary_cyto, 
                                                                    binary_cyto]), 0, -1)

    #create 3D array and rearrange shape to match an RGB image
    if return3D:
        binary_mask = numpy.moveaxis(numpy.asarray([shape_filtered_mask, 
                                                    shape_filtered_mask, 
                                                    shape_filtered_mask]), 0, -1)

    #return mask
    if return_cyto:
        return binary_mask.astype(int), binary_cyto.astype(int)

    else:
        return binary_mask.astype(int)


def maskEmpty(image_RGB, mask_val = 0.05, return3D = True, min_size = 150):

    """
    Method to remove white areas from RGB histology image.

    Parameters
    ----------

    image_RGB : 3D numpy array
        RGB image in the form [X, Y, C]

    mask_val : float:
        Value over which pixels will be masked out of hsv image in value space
    """

    hsv = rgb2hsv(image_RGB)

    binary_mask = (hsv[:,:,1] < mask_val).astype(int)

    labeled_mask = morph.label(binary_mask)

    labeled_mask = morph.remove_small_objects(labeled_mask, min_size =  min_size)

    labeled_mask = morph.remove_small_holes(labeled_mask)

    empty_mask = (labeled_mask < 1).astype(int)

    if return3D:
        empty_mask_3D = numpy.ones(image_RGB.shape, dtype = int)

        for i in range(empty_mask_3D.shape[-1]):
            empty_mask_3D[:,:,i] *= empty_mask

        return empty_mask_3D

    else:

        return empty_mask

def resetImage(corrected_image, original_image, mask_val = 0.1):
    input_dtype = corrected_image.dtype
    masked_image = corrected_image*fc.maskEmpty(original_image, mask_val=mask_val)
    
    final_image = numpy.where(masked_image[:,:,:] == 0, original_image,masked_image).astype(input_dtype)
    return final_image

def colorCorrection(image, reference, **kwargs):

    matched_image = ex.match_histogram(image, reference, multichannel = True)

    return matched_image


def singleChannel_falseColor(input_image, channelID = 's0', output_dtype = numpy.uint8):
    """depreciated
    single channel false coloring based on:
        Giacomelli et al., PLOS one 2016 doi:10.1371/journal.pone.0159337
    """
    
    beta_dict = {
                #nuclear consants
                's00' : {'K' : 0.017,
                             'R' : 0.544,
                             'G' : 1.000,
                             'B' : 0.050,
                             'thresh' : 50},
                             
                #cytoplasmic constants               
                's01' : {'K' : 0.008,
                              'R' : 0.300,
                              'G' : 1.000,
                              'B' : 0.860,
                              'thresh' : 500}}
                
    constants = beta_dict[channelID]
    
    RGB_image = numpy.zeros((input_image.shape[0],input_image.shape[1],3))
    
    #execute background subtraction
    input_image = preProcess(input_image,channelID)
    
    #assign RGB values
    R = numpy.exp(-constants['K']*constants['R']*input_image)
    G = numpy.exp(-constants['K']*constants['G']*input_image)
    B = numpy.exp(-constants['K']*constants['B']*input_image)
    
    #rescale to 8bit range
    RGB_image[:,:,0] = R*255
    RGB_image[:,:,1] = G*255
    RGB_image[:,:,2] = B*255
    
    return RGB_image.astype(output_dtype)


def combineFalseColoredChannels(nuclei, cyto, norm_factor = 255, output_dtype = numpy.uint8):
    """depreciated
    Use for combining false colored channels after single channel false color method
    """
    
    assert(cyto.shape == nuclei.shape)
 
    RGB_image = numpy.multiply(cyto/norm_factor,nuclei/norm_factor)
    RGB_image = numpy.multiply(RGB_image,norm_factor)
    
    return RGB_image.astype(output_dtype)
