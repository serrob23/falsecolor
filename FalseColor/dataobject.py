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
import tifffile as tf
import numpy
from pathos.multiprocessing import ProcessingPool
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
        """
        Loads list of tif images and returns as numpy array

        Parameters
        ----------

        file_list : list 
            List of tif filepaths to be read into memory

        Returns
        -------

        images : numpy array
            Image data read into memory
        """          
        file_list = sorted(file_list)
        images = []

        for item in file_list:
            images.append(tf.imread(item))

        return np.asarray(images)
            
    def loadH5(self, folder, dataID, 
                            channelIDs = ['s00', 's01'],
                            start_index = 0, stop_index = 0):

        """
        Parameters
        ----------

        folder : str or pathlike
            Folder to grab image data from

        dataID : int
            Resolution of data to grab from HDF5 file. Defaults to zero.

        channelIDs : list
            keys for data entry for HDF5
            
        start_index : int
            Index to begin reading data from, defaults to zero.

        stop_index : int
            Index to stop reading data from, if zero will continue through the entire dataset.

        Returns
        -------

        nuclei : 3D numpy array
            First channel image data from HDF5 file

        cyto : 3D numpy array
            Second channel image data from HDF5 file
        """

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

        """
        Sets up two channel H5 dataset with default key entries

        Parameters
        ----------

        folder : str or pathlike
            Folder to grab image data from, defaults to None. If None DataObject will use 
            self.directory

        dataID : int
            Resolution of data to grab from HDF5 file. Defaults to zero.

        channelIDs : list
            keys for data entry for HDF5

        start_index : int
            Index to begin reading data from, defaults to zero.

        stop_index : int
            Index to stop reading data from, if zero will continue through the entire dataset.


        Returns
        -------

        dataset : tuple
            Image data read from HDF5, in the order (channel1_data, channel2_data), where each entry
            is a numpy array of image data.

        """
        if folder:
            dataset = self.loadH5(folder, dataID=dataID, channelIDs=channelIDs)

        else:
            dataset = self.loadH5(self.directory, dataID=dataID, channelIDs=channelIDs)

        self.imageSet = numpy.asarray(dataset)
        
    
    def setupProcessing(self,ncpus):
        """
        Creates processing pool with specified ncpus for DataObject.

        Parameters
        ----------
        
        ncpus : int
            Number of cpu cores for processing pool

        """

        self.pool = ProcessingPool(ncpus=ncpus)
        
    def unloadPool(self):
        """
        Turns off processing pool.

        Parameters
        ----------

        Returns
        -------
        """
        self.pool = None
    
    def processImages(self,runnable_dict, imageSet, dtype = None):
            """
            Method to batch process multiple images simultaneously. Can process multiple 
            channels or one at a time. Method acts on Image data within data object. 

            Parameters
            -------
            
            runnable_dict : dict
            Dictionary with the following key, value pairs:

                'runnable' : method to run by processingpool.map

                'kwargs' : key word arguments for method

            imageSet : numpy array
                Image data of 3 or more dimmensions, should be in the shape 
                [[Z1, X1, Y1], [Z2, X2, Y2], ...etc].

            dtype : None or datatype
                Defaults to None type, if not none data will be returned as specified type. 

            Returns 
            -------

            processed_images : numpy array
                Images which have been procesed using the runnable_dict's method. 
            
            """
            if self.pool is None:
                self.setupProcessing(ncpus = 4)
            
            func,kwargs = runnable_dict['runnable'],runnable_dict['kwargs']
 
            processed_images = []

            if type(kwargs) == dict:
                processed_images.append(self.pool.map(func, *imageSet, **kwargs))
            
            else:
                processed_images.append(self.pool.map(func, *imageSet))

            if dtype is None:
                return numpy.asarray(processed_images)[0]
                            
            else:
                return numpy.asarray(processed_images, dtype = dtype)[0]