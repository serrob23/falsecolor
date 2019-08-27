
"""
Methods for H&E False coloring

Robert Serafin
8/12/2019

"""


import os
import skimage.morphology as morph
import skimage.filters as filt
import scipy.ndimage as nd
import numpy
import scipy.ndimage as nd
import skimage.feature as feat
from functools import partial
import glob
import time
import h5py as hp
import FalseColor_methods

def denoiseImage(img,*kwargs):
    """
    Denoise image by subtracting laplacian of gaussian
    """
    output_dtype = img.dtype
    img_gaus = nd.filters.gaussian_filter(img,sigma=3)
    img_log = nd.filters.laplace(img_gaus)#laplacian filter
    denoised_img = img - img_log # noise subtraction
    denoised_img[denoised_img < 0] = 0 # no negative pixels
    return denoised_img.astype(output_dtype)

def padImages(img,*kwargs):
    hoff,voff = 35,3
    output_dtype = img.dtype
    newimg = numpy.pad(img,((voff,voff),(hoff,hoff)),'constant',constant_values = 0)
    return newimg.astype(output_dtype)

def unpadImages(img,*kwargs):
    hoff,voff = 35,3
    output_dtype = img.dtype
    newimg = img[voff:-voff:1,hoff:-hoff:1]
    return newimg.astype(output_dtype)

def singleChannel_falseColor(input_image,channelID='s0',output_dtype = numpy.uint8):
    """
    single channel false coloring based on:
        Giacomelli et al., PLOS one 2016 doi:10.1371/journal.pone.0159337
    """
    
    beta_dict = {'cytoplasm' : {'K' : 0.008,
                              'R' : 0.300,
                              'G' : 1.000,
                              'B' : 0.860,
                              'thresh' : 500},
                's01' : {'K' : 0.008,
                              'R' : 0.300,
                              'G' : 1.000,
                              'B' : 0.860,
                              'thresh' : 500},
                 'nuclei' : {'K' : 0.017,
                             'R' : 0.544,
                             'G' : 1.000,
                             'B' : 0.050,
                             'thresh' : 50},
                's00' : {'K' : 0.017,
                             'R' : 0.544,
                             'G' : 1.000,
                             'B' : 0.050,
                             'thresh' : 50}}

    constants = beta_dict[channelID]
    
    RGB_image = numpy.zeros((input_image.shape[0],input_image.shape[1],3))
    
    input_image = preProcess(input_image,channelID)
    
    R = numpy.exp(-constants['K']*constants['R']*input_image)
    G = numpy.exp(-constants['K']*constants['G']*input_image)
    B = numpy.exp(-constants['K']*constants['B']*input_image)
    
    RGB_image[:,:,0] = R*255
    RGB_image[:,:,1] = G*255
    RGB_image[:,:,2] = B*255
    
    return RGB_image.astype(output_dtype)

def combineFalseColoredChannels(input_image,norm_factor = 255,output_dtype = numpy.uint8):

    nuclei,cytoplasm = input_image[0],input_image[1]
    
    assert(cytoplasm.shape == nuclei.shape)
 
    RGB_image = numpy.multiply(cytoplasm/norm_factor,nuclei/norm_factor)
    RGB_image = numpy.multiply(RGB_image,norm_factor)
    
    return RGB_image.astype(output_dtype)
    
    

def falseColor(imageSet,channelIDs=['s00','s01'],output_dtype=numpy.uint8):
    """
    expects input imageSet data to be structured in the same way as in the FCdataobject

    """
    # print(type(imageSet))

    beta_dict = {'cytoplasm' : {'K' : 0.008,
                              'R' : 0.300,
                              'G' : 1.000,
                              'B' : 0.860,
                              'thresh' : 500},
                's01' : {'K' : 0.008,
                              'R' : 0.300,
                              'G' : 1.000,
                              'B' : 0.860,
                              'thresh' : 500},
                 'nuclei' : {'K' : 0.017,
                             'R' : 0.544,
                             'G' : 1.000,
                             'B' : 0.050,
                             'thresh' : 50},
                's00' : {'K' : 0.017,
                             'R' : 0.544,
                             'G' : 1.000,
                             'B' : 0.050,
                             'thresh' : 50}}

    constants_nuclei = beta_dict[channelIDs[0]]

    k_nuclei = constants_nuclei['K']

    constants_cyto = beta_dict[channelIDs[1]]

    k_cytoplasm= constants_cyto['K']
    
    
    
    nuclei = preProcess(imageSet[:,:,0],channelIDs[0])
    cyto = preProcess(imageSet[:,:,1],channelIDs[1])

    RGB_image = numpy.zeros((nuclei.shape[0],nuclei.shape[1],3))
    
    
    
    R = numpy.multiply(numpy.exp(-constants_cyto['R']*k_cytoplasm*cyto),
                                    numpy.exp(-constants_nuclei['R']*k_nuclei*nuclei))

    G = numpy.multiply(numpy.exp(-constants_cyto['G']*k_cytoplasm*cyto),
                                    numpy.exp(-constants_nuclei['G']*k_nuclei*nuclei))

    B = numpy.multiply(numpy.exp(-constants_cyto['B']*k_cytoplasm*cyto),
                                    numpy.exp(-constants_nuclei['B']*k_nuclei*nuclei))

    RGB_image[:,:,0] = (R*255)
    RGB_image[:,:,1] = (G*255)
    RGB_image[:,:,2] = (B*255)
    return RGB_image.astype(output_dtype)

def preProcess(images, channelID, nuclei_thresh = 50, cyto_thresh = 500):

    channel_parameters = {'s00' : {'thresh' : 50},
                          's01' : {'thresh' : 50}}


    #parameters for background subtraction
    thresh = channel_parameters[channelID]['thresh']
    images -= thresh

    images = numpy.power(images,0.85)

    # images = numpy.where(images>thresh,images,0)

    image_mean = numpy.mean(images[images>thresh])*8

    processed_images = images*(65535/image_mean)*(255/65535)

    processed_images[processed_images < 0] = 0

    return processed_images


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

def gaussianSmoothing(images,sigma = 3.0):
    return nd.filters.gaussian_filter(images,(sigma,sigma))

def medianBlur(images):
    return nd.filters.median_filter(images,1)