"""
#===============================================================================
# 
#  License: GPL
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
11/4/2019

"""


import os
import scipy.ndimage as nd
import skimage.filters as filt
import skimage.exposure as ex
import skimage.util as util
import cv2
import numpy
import h5py as hp
from numba import cuda
import math
    
def falseColor(imageSet, channelIDs=['s00','s01'], 
                            output_dtype=numpy.uint8):
    """
    imageSet : 3D numpy array
        dimmensions are [X,Y,C]
        for use with process images in FCdataobject

    false coloring based on:
        Giacomelli et al., PLOS one 2016 doi:10.1371/journal.pone.0159337

    channelIDs = list
        keys to grab settings from beta_dict
        defaults: s00 : nuclei
                  s01 : cyto

    output_dtype : numpy.uint8
        output datatype for final RGB image

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

    nuclei = imageSet[:,:,0].astype(float)
    constants_nuclei = settings['nuclei']
    k_nuclei = beta_dict['K_nuclei']

    cyto = imageSet[:,:,1].astype(float)
    constants_cyto = settings['cyto']
    k_cytoplasm= beta_dict['K_cyto']
    
    #execute background subtraction
    nuc_threshold = getBackgroundLevels(nuclei)[1]
    nuclei = preProcess(nuclei, threshold = nuc_threshold)
    
    cyto_threshold = getBackgroundLevels(cyto)[1]
    cyto = preProcess(cyto, threshold = cyto_threshold)

    RGB_image = numpy.zeros((3,nuclei.shape[0],nuclei.shape[1]))

    #iterate throough RGB constants and execute image multiplication
    for i in range(len(RGB_image)):
        RGB_image[i] = 255*numpy.multiply(numpy.exp(-constants_cyto[i]*k_cytoplasm*cyto),
                                        numpy.exp(-constants_nuclei[i]*k_nuclei*nuclei))

    #reshape to [X,Y,C]
    RGB_image = numpy.moveaxis(RGB_image,0,-1)

    #rescale to 8bit range
    return RGB_image.astype(output_dtype)

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
    k_nuclei = 0.85
    nuclei_RGBsettings = [0.25*k_nuclei, 0.37*k_nuclei, 0.1*k_nuclei]
    cyto_RGB_settings = [0.05*k_cyto, 1.0*k_cyto, 0.54*k_cyto]

    settings_dict = {'nuclei':nuclei_RGBsettings,'cyto':cyto_RGB_settings}
    return settings_dict

def preProcess(image, threshold = 50):
    """
    Method used for background subtracting data with a fixed value

    image : 2d numpy array
        image for processing

    threshold : int
        background level to subtract
    """

    #background subtraction
    image -= threshold

    #no negative values
    image[image < 0] = 0

    #calculate normalization factor
    image = numpy.power(image,0.85)
    image_mean = numpy.mean(image[image>threshold])*8

    #convert into 8bit range
    processed_image = image*(65535/image_mean)*(255/65535)

    return processed_image

@cuda.jit #direct GPU compiling
def rapid_preProcess(image,background,norm_factor,output):
    """Background subtraction optimized for gpu

    image : 2d numpy array, dtype = int16
        image for background subtraction

    background : int
        constant for subtraction

    norm_factor : int
        empirically determaned constant for normalization after subtraction

    output : 2d numpy array
        numpy array of zeros for gpu to assign values to
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

@cuda.jit #direct GPU compiling
def rapid_getRGBframe(nuclei, cyto, output, nuc_settings,
                        cyto_settings, k_nuclei, k_cyto):
    #TODO: implement array base normalization
    """
    nuclei : numpy array
        nuclear channel image
        already pre processed

    cyto : numpy array
        cyto channel image
        already pre processed
        
    nuc_settings : float
        RGB constant for nuclear channel
    
    cyto_settings : float
        RGB constant for cyto channel
    """
    row,col = cuda.grid(2)

    #iterate through image and assign pixel values
    if row < output.shape[0] and col < output.shape[1]:
        tmp = nuclei[row,col]*nuc_settings*k_nuclei + cyto[row,col]*cyto_settings*k_cyto
        output[row,col] = 255*math.exp(-1*tmp)

@cuda.jit
def rapidFieldDivision(image,flat_field,output):
    """
    used for falseColoring when flat field has been calculated

    image : numpy array written to GPU

    flat_field : numpy array written to GPU

    output : numpy array written to GPU

    """
    row,col = cuda.grid(2)

    if row < output.shape[0] and col < output.shape[1]:
        tmp = image[row,col]/flat_field[row,col]
        output[row,col] = tmp

def rapidFalseColor(nuclei, cyto, nuc_settings, cyto_settings,
                   TPB=(32,32) , nuc_normfactor = 8500, cyto_normfactor=3000,
                   nuc_background = 50, cyto_background = 50,
                   run_normalization = False):
    """
    nuclei : numpy array
        nuclear channel image
        
    cyto : numpy array
        cyto channel image
        
    nuc_settings : list
        settings of RGB constants for nuclear channel
    
    cyto_settings : list
        settings of RGB constants for cyto channel

    nuc_normfactor : int or array
        defaults to empirically determined constant for background subtraction,
        else should be a numpy array representing the flat field

    cyto_normfactor : int or array
        defaults to empirically determined constant for background subtraction,
        else should be a numpy array representing the flat field

    nuc_background : int or float
        defaults to 50, background threshold for subtraction

    cyt_background : int or float
        defaults to 50, background threshold for subtraction
        
    TPB : tuple (int,int)
        THREADS PER BLOCK: (x_threads,y_threads)
        used for GPU threads
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
    if run_normalization:
        nuc_normfactor = numpy.ascontiguousarray(nuc_normfactor)
        nuc_norm_mem = cuda.to_device(nuc_normfactor)
        rapidFieldDivision[blockspergrid,TPB](nuc_global_mem,nuc_norm_mem,pre_nuc_output)

    #otherwise use standard background subtraction
    else:
        k_nuclei = 0.08
        nuc_background = getBackgroundLevels(nuclei)[1]
        rapid_preProcess[blockspergrid,TPB](nuc_global_mem,nuc_background,
                                                nuc_normfactor,pre_nuc_output)
    
    #allocate memory for background subtraction
    cyto = numpy.ascontiguousarray(cyto)
    pre_cyto_output = cuda.to_device(numpy.zeros(cyto.shape))
    cyto_global_mem = cuda.to_device(cyto)

    #run background subtraction or normalization for cyto

    #use flat fielding
    if run_normalization:
        cyto_normfactor = numpy.ascontiguousarray(cyto_normfactor)
        cyto_norm_mem = cuda.to_device(cyto_normfactor)
        rapidFieldDivision[blockspergrid,TPB](cyto_global_mem,cyto_norm_mem,pre_cyto_output)

    # otherwise use standard background subtraction
    else:
        k_cyto = 0.012
        cyto_background = getBackgroundLevels(cyto)[1]
        rapid_preProcess[blockspergrid,TPB](cyto_global_mem,cyto_background,
                                                cyto_normfactor,pre_cyto_output)
    
    #create output array to iterate through
    RGB_image = numpy.zeros((3,nuclei.shape[0],nuclei.shape[1]),dtype = numpy.int8) 

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

@cuda.jit
def Convolve2d(image,kernel,output):

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
    #create kernels to amplify edges
    hkernel = numpy.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    vkernel = numpy.array([[1,0,-1],[1,0,-1],[1,0,-1]])

    #set grid/threads for GPU
    blocks = (32,32)
    grid = (input_image.shape[0]//blocks[0] + 1, input_image.shape[1]//blocks[1] + 1)

    #run convolution
    voutput = numpy.zeros(input_image.shape,dtype=numpy.float64)
    houtput = numpy.zeros(input_image.shape,dtype=numpy.float64)
    Convolve2d[grid,blocks](input_image,vkernel,voutput)
    Convolve2d[grid,blocks](input_image,hkernel,houtput)

    #calculate final result
    final_image = input_image + 0.5*numpy.sqrt(voutput**2 + houtput**2)
    
    return final_image

def getBackgroundLevels(image, threshold = 50):

    image_DS = numpy.sort(image,axis=None)

    foreground_vals = image_DS[numpy.where(image_DS > threshold)]

    hi_val = foreground_vals[int(numpy.round(len(foreground_vals)*0.95))]

    background = hi_val/5

    return hi_val,background

def getFlatField(image,tileSize=256,block_size = 16):
    #returns downsample flat field of image data and calculated background levels

    midrange,background = getBackgroundLevels(image)
    
    rows_max = int(numpy.floor(image.shape[0]/block_size)*block_size)
    cols_max = int(numpy.floor(image.shape[2]/block_size)*block_size)
    stacks_max = int(numpy.floor(image.shape[1]/block_size)*block_size)


    rows = numpy.arange(0, rows_max+int(tileSize/block_size), int(tileSize/block_size))
    cols = numpy.arange(0, cols_max+int(tileSize/block_size), int(tileSize/block_size))
    stacks = numpy.arange(0, stacks_max+int(tileSize/block_size), int(tileSize/block_size))
    
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
