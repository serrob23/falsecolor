"""
False Coloring Methods
By: Robert Serafin

8/12/2019


"""

import os,glob,json
import numpy as np
import skimage.feature as feat
import skimage.morphology as morph
import skimage.filters as filt
import tifffile as tif
import xmltodict


def denoiseImage(img,*kwargs):
    """
    Denoise image by subtracting laplacian of gaussian
    """
    output_dtype = img.dtype
    img_gaus = filt.gaussian(img,sigma=3)
    img_log = filt.laplace(img_gaus) #laplacian filter
    denoised_img = img - img_log # noise subtraction
    denoised_img[denoised_img < 0] = 0 # no negative pixels
    return denoised_img.astype(output_dtype)

def padImages(img,*kwargs):
    hoff,voff = 35,3
    output_dtype = img.dtype
    newimg = np.pad(img,((voff,voff),(hoff,hoff)),'constant',constant_values = 0)
    return newimg.astype(output_dtype)

def unpadImages(img,*kwargs):
    hoff,voff = 35,3
    output_dtype = img.dtype
    newimg = img[voff:-voff:1,hoff:-hoff:1]
    return newimg.astype(output_dtype)

def singleChannel_falseColor(input_image,channelID,output_dtype = np.uint8):
    """
    single channel false coloring based on:
        Giacomelli et al., PLOS one 2016 doi:10.1371/journal.pone.0159337
    """
    
    beta_dict = {'cytoplasm' : {'K' : 0.008,
                              'R' : 0.300,
                              'G' : 1.000,
                              'B' : 0.860},
                 'nuclei' : {'K' : 0.017,
                             'R' : 0.544,
                             'G' : 1.000,
                             'B' : 0.050}}
    
    
    RGB_image = np.zeros((input_image.shape[0],input_image.shape[1],3))
    
    
    constants = beta_dict[channelID]
    
    R = np.exp(-constants['K']*constants['R']*input_image)
    G = np.exp(-constants['K']*constants['G']*input_image)
    B = np.exp(-constants['K']*constants['B']*input_image)
    
    RGB_image[:,:,0] = R*255
    RGB_image[:,:,1] = G*255
    RGB_image[:,:,2] = B*255
    
    return RGB_image.astype(output_dtype)

def combineFalseColoredChannels(cytoplasm,nuclei,norm_factor = 255,output_dtype = np.uint8):
    
    assert(cytoplasm.shape == nuclei.shape)
 
    RGB_image = np.multiply(cytoplasm/norm_factor,nuclei/norm_factor)
    RGB_image = np.multiply(RGB_image,norm_factor)
    
    return RGB_image.astype(output_dtype)
    
    

def dualFalseColor(cytosol,nuclei):
    cytosol = np.power(cytosol,0.85)
    nuclei = np.power(nuclei,0.85)

    k_cytoplasm=0.008
    k_nuclei=0.017
    
    beta1=0.860
    beta2=0.050
    
    beta3=1.0
    beta4=1.0
    
    beta5=0.300
    beta6=0.544
    
    
    R = np.multiply(np.exp(-beta5*k_cytoplasm*cytosol),np.exp(-beta6*k_nuclei*nuclei))
    G = np.multiply(np.exp(-beta3*k_cytoplasm*cytosol),np.exp(-beta4*k_nuclei*nuclei))
    B = np.multiply(np.exp(-beta1*k_cytoplasm*cytosol),np.exp(-beta2*k_nuclei*nuclei))
    image = np.zeros((cytosol.shape[0],cytosol.shape[1],3))
    image[:,:,0] = (R*255)
    image[:,:,1] = (G*255)
    image[:,:,2] = (B*255)
    image = image.astype(np.uint8)
    return image.astype(np.uint8)
    