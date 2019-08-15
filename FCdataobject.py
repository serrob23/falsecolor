"""
DataObject for H&E False coloring

Robert Serafin
8/14/2019

"""

import os
import tifffile as tf
import skimage.morphology as morph
import skimage.filters as filt
import numpy
import scipy.ndimage as nd
import skimage.feature as feat
from pathos.multiprocessing import ProcessingPool
from functools import partial
import glob
import time
import h5py as hp
import FalseColor_methods


class DataObject(object):
    def __init__(self,directory, imageSet = None, channel_IDs = None, setupPool = False):
        """
        Object to store image data in a convienient way for batch processing
        
        Attributes
        ----------
        
        directory : string
            Base directory where image data is stored
            
        imageSet : dict #TODO: convert to zipped array for tif loading
            default None, if passed the directory structure should be 
            
            imageSet = {'channel_name' : {
                                        'data' : numpy.array, #image data
                                        'files : list #list of file names ending in
                                        }
                        ....}
        
        channel_IDs : list
            list of channel names i.e ['channel1','channel2'...]
            used to sort data
        
        setupPool : bool
            setup processing pool
        
        
        """
        
        # object base directory
        self.directory = directory
        
        
        #channel IDs are expected to be a list
        if channel_IDs:
            self.channel_IDs = channel_IDs
        else:
            self.channel_IDs = []
        
        
        if imageSet is not None:
            self.imageSet = imageSet
        else:
            self.imageSet = {}
        
        
        if setupPool:
            self.setupProcessing(ncpus = 2)
        else:
            self.unloadPool()
                
    
    def loadTifImages(self,file_list,image_size,channel_ID):            
        try:
            assert(type(image_size == tuple))
            assert(len(image_size) > 0)
            file_list = sorted(file_list)
            file_names = []
            images = numpy.zeros((len(file_list),image_size[0],image_size[1]))

            for i,z in enumerate(file_list):
                images[i] = tf.imread(z)
                file_names.append(z.split(os.sep)[-1])
            
            if channel_ID not in self.imageSet.keys():
                self.imageSet[channel_ID] = {}
            
            self.imageSet[channel_ID]['data'] = images
            self.imageSet[channel_ID]['files'] = file_names
        
        except AssertionError:
            print('Image_size must be tuple of form (m,n) where m and n are integers')
            
    def loadH5(self,folder,image_size = None, channel_ID = None):
        
        data_name = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('h5')]
        
        H5_dataset = hp.File(data_name[0],'r')

        print(list(H5_dataset.keys()))
        
        return H5_dataset #or whatever the data is actually called in the xml file
        
    def setupH5data(self,folder=None,dataID = 0):

        self.channel_IDs = ['s00','s01']

        if folder:
            dataset = self.loadH5(folder)

        else:
            dataset = self.loadH5(self.directory)

        #Create imageSet as a 4D array, from the loaded dataset
        self.imageSet = numpy.stack((dataset['t00000'][self.channel_IDs[0]][str(dataID)]['cells'],
            dataset['t00000'][self.channel_IDs[1]][str(dataID)]['cells']),axis=-1)
        print(self.imageSet.shape)
    
    def setupProcessing(self,ncpus):
        self.pool = ProcessingPool(ncpus=ncpus)
        
    def unloadPool(self):
        self.pool = None
    
    def processImages(self,runnable_dict, imageSet = None, singleSet = True):
            """
            Purpose
            -------
            Method to batch process multiple images simultaneously. Can process multiple 
            channels or one at a time. Method acts on Image data within data object. 
            
            Attributes
            ----------
            
            runnable_dict : dict
                Currently should have one key 'runnable' which is mapped to a method to be run
            
            channel_IDs : list
                list of strings which correspond to keys in imageSet dictonary
            
            """
            if self.pool is None:
                self.setupProcessing(ncpus=4)
            
            runnable,*args = runnable_dict['runnable'],runnable_dict['args']
            print(runnable,*args,imageSet.shape)
            method = partial(runnable,*args)
            
            
            if singleSet:
                processed_images = {}
                for chan in channel_IDs:
                    print(chan)
                    
                    #data set to be acted upon
                    images = self.imageSet[chan]['data'] 
                    
                    #allows setting of keyword arguments in runnable beforehand as one function for map
                    method = partial(runnable_dict['runnable'],channelID = chan)
                    
                    #map runnable, processed images will be a dictonary with channel IDs as keys
                    #each key is mapped to the corresponding image array
                    processed_images[chan] = numpy.asarray(self.pool.map(method,images))

                
                processed_images = numpy.stack((processed_images[channel_IDs[0]],
                    processed_images[channel_IDs[1]]),axis = -1)

                return processed_images

            elif imageSet is not None:
                processed_images = []
                method = partial(runnable,*args)

                processed_images.append(self.pool.map(method,imageSet))

                return numpy.asarray(processed_images)
