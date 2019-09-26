
"""
Methods for H&E False coloring

Robert Serafin
9/11/2019

"""


import os
import scipy.ndimage as nd
import skimage.feature as feat
import skimage.morphology as morph
import skimage.filters as filt
import skimage.exposure as ex
import skimage.util as uitl
import scipy.ndimage as nd
import cv2
import numpy
import h5py as hp
from numba import cuda
import math
import copy
import tensorflow as tf
from astropy.convolution import Gaussian2DKernel
    
def falseColor(imageSet, channelIDs=['s00','s01'], output_dtype=numpy.uint8):
    """
    imageSet : 3D numpy array
        dimmensions are [X,Y,C]

    false coloring based on:
        Giacomelli et al., PLOS one 2016 doi:10.1371/journal.pone.0159337

    """
    beta_dict = {
                #constants for nuclear channel
                's00' : {'K' : 0.017,
                             'R' : 0.544,
                             'G' : 1.000,
                             'B' : 0.050,
                             'thresh' : 50},

                #constants for cytoplasmic channel
                's01' : {'K' : 0.008,
                              'R' : 0.30,
                              'G' : 1.0,
                              'B' : 0.860,
                              'thresh' : 500}
                              }

    #assign constants for each channel
    constants_nuclei = beta_dict[channelIDs[0]]
    k_nuclei = constants_nuclei['K']

    constants_cyto = beta_dict[channelIDs[1]]
    k_cytoplasm= constants_cyto['K']
    
    #execute background subtraction
    nuclei = preProcess(imageSet[:,:,0])
    cyto = preProcess(imageSet[:,:,1])

    RGB_image = numpy.zeros((nuclei.shape[0],nuclei.shape[1],3))

    #assign RGB values from grayscale images
    R = numpy.multiply(numpy.exp(-constants_cyto['R']*k_cytoplasm*cyto),
                                    numpy.exp(-constants_nuclei['R']*k_nuclei*nuclei))

    G = numpy.multiply(numpy.exp(-constants_cyto['G']*k_cytoplasm*cyto),
                                    numpy.exp(-constants_nuclei['G']*k_nuclei*nuclei))

    B = numpy.multiply(numpy.exp(-constants_cyto['B']*k_cytoplasm*cyto),
                                    numpy.exp(-constants_nuclei['B']*k_nuclei*nuclei))

    #rescale to 8bit range
    RGB_image[:,:,0] = (R*255)
    RGB_image[:,:,1] = (G*255)
    RGB_image[:,:,2] = (B*255)
    return RGB_image.astype(output_dtype)

def preProcess(image, thresh = 50):
    """
    image : 2d numpy array
        image for processing

    threshold : int
        background level to subtract
    """

    #background subtraction
    image -= thresh

    #no negative values
    image[image < 0] = 0

    #calculate normalization factor
    image = numpy.power(image,0.85)
    image_mean = numpy.mean(image[image>thresh])*8

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
        tmp = (image[row,col] - background)**0.85

        #normalize to 8bit range
        tmp = image[row,col]*(65535/norm_factor)*(255/65535)
        output[row,col] =tmp

        #remove negative values
        if tmp < 0:
            output[row,col] = 0

@cuda.jit #direct GPU compiling
def rapid_getRGBframe(nuclei,cyto,output,nuc_settings,
                        cyto_settings):
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
        tmp = nuclei[row,col]*nuc_settings + cyto[row,col]*cyto_settings
        output[row,col] = 255*math.exp(-1*tmp)

@cuda.jit
def rapidFieldDivision(image,flat_field,output):
    """
    used for falseColoring when flat fielding has been done

    image : numpy array

    flat_field : numpy array

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

    #create blockgrid for gpu
    blockspergrid_x = int(math.ceil(nuclei.shape[0] / TPB[0]))
    blockspergrid_y = int(math.ceil(nuclei.shape[1] / TPB[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    #allocate memory for background subtraction
    nuclei = numpy.ascontiguousarray(nuclei)
    pre_nuc_output = cuda.to_device(numpy.zeros(nuclei.shape))
    nuc_global_mem = cuda.to_device(nuclei)

    #run background subtraction or normalization for nuclei
    if run_normalization:
        nuc_normfactor = numpy.ascontiguousarray(nuc_normfactor)
        nuc_norm_mem = cuda.to_device(nuc_normfactor)
        rapidFieldDivision[blockspergrid,TPB](nuc_global_mem,nuc_norm_mem,pre_nuc_output)

    else:
        rapid_preProcess[blockspergrid,TPB](nuc_global_mem,nuc_background,
                                                nuc_normfactor,pre_nuc_output)
    
    #allocate memory for background subtraction
    cyto = numpy.ascontiguousarray(cyto)
    pre_cyto_output = cuda.to_device(numpy.zeros(cyto.shape))
    cyto_global_mem = cuda.to_device(cyto)

    #run background subtraction for cyto
    if run_normalization:
        cyto_normfactor = numpy.ascontiguousarray(nuc_normfactor)
        cyto_norm_mem = cuda.to_device(cyto_normfactor)
        rapidFieldDivision[blockspergrid,TPB](cyto_global_mem,cyto_norm_mem,pre_cyto_output)

    else:
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
                                                nuc_settings[i],cyto_settings[i])
        
        RGB_image[i] = output_global.copy_to_host()

    #reorder array to dimmensional form [X,Y,C]
    RGB_image = numpy.moveaxis(RGB_image,0,-1)
    return RGB_image.astype(numpy.uint8)

def sharpenImage(input_image,alpha = 0.5):
    #create kernels to amplify edges
    horizontal = numpy.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    vertical = numpy.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    
    blocks = (32,32)
    grid = (input_image.shape[0] // blocks[0] + 1, input_image.shape[1] // blocks[1] + 1)




    #create output arrays for each convolution
    h_output = numpy.zeros(input_image.shape, dtype = numpy.float64)
    h_output_mem = cuda.to_device(h_output)
    input_global_mem = cuda.to_device(numpy.ascontiguousarray(input_image))

    print('horizontal')
    convolve2D[grid,blocks](input_global_mem, horizontal, h_output_mem)

    v_output = numpy.zeros(input_image.shape, dtype = numpy.float64)
    v_output_mem = cuda.to_device(numpy.ascontiguousarray(v_output))
    print('vertical')
    convolve2d[grid,blocks](input_global_mem,vertical,v_output_mem)

    #cuda accelerated convolution of input image with each kernel



    #final sharpening of image
    final_image = input_image + alpha*numpy.sqrt(h_output**2 + v_output**2)
    
    return final_image

def gaussianBlur(input_image,sigma):
    kernel = numpy.asarray(Gaussian2DKernel(sigma))

    blocks = (32,32)
    grid = (input_image.shape[0] // blocks[0] + 1, input_image.shape[1] // blocks[1] + 1)

    output_image = numpy.zeros(input_image.shape)

    convolve2D[grid,blocks](copy.deepcopy(input_image), kernel, output_image)

    return output_image


@cuda.jit
def convolve2D(image,kernel,output_image):
    #create iterators
    row,col = cuda.grid(2)

    #define iterator boundaries
    image_rows, image_cols = image.shape

    #ignore threads outside image
    if (row >= image_rows) or (col >= image_cols):
        return

    #window size for computing result at i,j 
    delta_R = kernel.shape[0]//2
    delta_C = kernel.shape[1]//2

    tmp = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # retult at i,j is:
            # sum(i,j) mask[i,j]*image[row-i+delta_R,col-j+delta_C]
            # where i,j iterate through the kernel
            row_i = row - i + delta_R
            col_j = col - j + delta_C

            #only compute for indicies within image boundary
            if (row_i >= 0) and (row_i < image_rows):
                if (col_j >= 0) and (col_j < image_cols):
                    tmp += kernel[i,j]*image[row_i,col_j]

    output_image[row,col] = tmp

def singleChannel_falseColor(input_image, channelID = 's0', output_dtype = numpy.uint8):
    """
    single channel false coloring based on:
        Giacomelli et al., PLOS one 2016 doi:10.1371/journal.pone.0159337
    """
    
    beta_dict = {
                #nuclear consants
                's01' : {'K' : 0.008,
                              'R' : 0.300,
                              'G' : 1.000,
                              'B' : 0.860,
                              'thresh' : 500},
                #cytoplasmic constants
                's00' : {'K' : 0.017,
                             'R' : 0.544,
                             'G' : 1.000,
                             'B' : 0.050,
                             'thresh' : 50}}

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

def combineFalseColoredChannels(input_image, norm_factor = 255, output_dtype = numpy.uint8):

    nuclei,cytoplasm = input_image[0],input_image[1]
    
    assert(cytoplasm.shape == nuclei.shape)
 
    RGB_image = numpy.multiply(cytoplasm/norm_factor,nuclei/norm_factor)
    RGB_image = numpy.multiply(RGB_image,norm_factor)
    
    return RGB_image.astype(output_dtype)

def adaptiveBinary(images, blocksize = 15,offset = 0):
    if len(images.shape) == 2:
        filtered_img = medianBlur(images)
        binary_img = images > filt.threshold_local(filtered_img,blocksize,offset = offset)
    else:
        binary_img = numpy.zeros(images.T.shape)
        for i,z in enumerate(images.T):
            filtered_z = gaussianSmoothing(medianBlur(z))
            binary_img[i] = z > filt.threshold_local(filtered_z,blocksize,offset=offset)
        binary_img = binary_img.T
    return numpy.asarray(binary_img,dtype =int)
    

def medianBlur(images):
    return nd.filters.median_filter(images,1)

def tophat_filter(image):
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
    return cv2.morphologyEx(image,cv2.MORPH_TOPHAT,el)

def filter_and_equalize(Image,kernel_size = 204):
    z = tophat_filter(Image)
    z = ex.equalize_adapthist(z,kernel_size=kernel_size)
    return util.img_as_uint(z)

def denoiseImage(img):
    """
    Denoise image by subtracting laplacian of gaussian
    """
    output_dtype = img.dtype
    img_gaus = nd.filters.gaussian_filter(img,sigma=3)
    img_log = nd.filters.laplace(img_gaus)#laplacian filter
    denoised_img = img - img_log # noise subtraction
    denoised_img[denoised_img < 0] = 0 # no negative pixels
    return denoised_img.astype(output_dtype)

def getBackgroundLevels(imageSet, threshold = 50):

    image_DS = numpy.sort(imageSet,axis=None)

    foreground_vals = image_DS[numpy.where(image_DS > threshold)]

    hi_val = foreground_vals[int(numpy.round(len(foreground_vals)*0.95))]

    background = hi_val/5

    return hi_val,background

def getFlatField(image,tileSize=256):
    #returns flat field of image and calculated background levels
    midrange,background = getBackgroundLevels(image)
    
    rows_max = int(numpy.floor(image.shape[0]/16)*16)
    cols_max = int(numpy.floor(image.shape[2]/16)*16)
    stacks_max = int(numpy.floor(image.shape[1]/16)*16)

    rows = numpy.arange(0, rows_max+int(tileSize/16), int(tileSize/16))
    cols = numpy.arange(0, cols_max+int(tileSize/16), int(tileSize/16))
    stacks = numpy.arange(0, stacks_max+int(tileSize/16), int(tileSize/16))
    
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




