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
import tifffile as tf
import numpy
from pathos.multiprocessing import ProcessingPool
from functools import partial
import h5py as hp
import FalseColor.Color as fc

class DataObject(object):
    def __init__(self, directory, imageSet = None,
                 setupPool = False, ncpus = 2, tissue_type = 'Default'):
        """
        Object to store image data in a convienient way for batch processing
        
        Attributes
        ----------
        
        directory : string or pathlike
            Base directory where image data is stored
            
        imageSet : 3d numpy array
            images for processing
        
        setupPool : bool
            setup processing pool
        
        """
        
        # object base directory
        self.directory = directory

        #object data to process
        self.imageSet = imageSet
        
        #dedicated cpus
        if setupPool:
            self.setupProcessing(ncpus = ncpus)
        else:
            self.unloadPool()

        #Tissue type for RGB settings
        self.tissue = tissue_type
                
    
    def loadTifImages(self, file_list):            
        file_list = sorted(file_list)
        images = []

        for i in range(len(file_list)):
            images.append(tf.imread(file_list[i]))

        return np.asarray(images)
            
    def loadH5(self, folder, dataID, 
                            channelIDs = ['s00', 's01'],
                            start_index = 0, stop_index = 0):

        data_name = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('h5')]
        
        if start_index == stop_index:

            with hp.File(data_name[0], 'r') as f:
                nuclei = f['t00000'][channelIDs[0]][str(dataID)]['cells'][:]
                cyto = f['t00000'][channelIDs[1]][str(dataID)]['cells'][:]
            f.close()

        else:

            with hp.File(data_name[0], 'r') as f:
                nuclei = f['t00000'][channelIDs[0]][str(dataID)]['cells'][start_index:stop_index]
                cyto = f['t00000'][channelIDs[0]][str(dataID)]['cells'][start_index:stop_index]
            f.close()

        return nuclei, cyto
        
    def setupH5data(self, folder = None, dataID = 0,
                            channelIDs = ['s00','s01'],
                            start_index = 0, stop_index = 0):

        if folder:
            dataset = self.loadH5(folder, dataID=dataID, channelIDs=channelIDs)

        else:
            dataset = self.loadH5(self.directory, dataID=dataID, channelIDs=channelIDs)

        #Create imageSet as a 4D array, from the loaded dataset
        self.imageSet = numpy.stack((dataset[0], dataset[1]),axis=-1)
        
        print(self.imageSet.shape)
    
    def setupProcessing(self,ncpus):
        self.pool = ProcessingPool(ncpus=ncpus)
        
    def unloadPool(self):
        self.pool = None
    
    def processImages(self,runnable_dict, imageSet, dtype = None):
            """
            Purpose
            -------
            Method to batch process multiple images simultaneously. Can process multiple 
            channels or one at a time. Method acts on Image data within data object. 
            
            Attributes
            ----------
            
            runnable_dict : dict
                Currently should have one key 'runnable' which is mapped to a method to be run
                the other key 'kwargs' are the inputs to the method which are different than
                the method's default parameters
            
            """
            if self.pool is None:
                self.setupProcessing(ncpus = 4)
            
            func,kwargs = runnable_dict['runnable'],runnable_dict['kwargs']
            print(func, kwargs, imageSet.shape)
 
            processed_images = []
            # method = partial(runnable,**kwargs)

            if type(kwargs) == dict:
                processed_images.append(self.pool.map(func, imageSet, **kwargs))
            
            else:
                processed_images.append(self.pool.map(func, imageSet))

            if dtype is None:
                return numpy.asarray(processed_images)
            
            else:
                return numpy.asarray(processed_images, dtype = dtype)