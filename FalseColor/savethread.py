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
import tifffile as tif
import numpy
from skimage import io
import time
from FalseColor import Process


def saveProcess(queue):
    """
    Parameters
    ----------

    queue : multiprocessing queue

        message : result of queue.get(), has the following properties in this order:

                path : str or pathlike
                    top level storage directory for data
                
                folder : str
                    specific dir to save data in

                filename : str
                    "file.tif" filename for data

                data : numpy array
                    image to save

                do_stats : bool, False
                    whether to save RGB statistics, defaults to False

                token : None or str
                    token will be a str when thread stop is called
    
    Returns
    -------
    """
    while True:

        message = queue.get()

        if message[-1] is not None:
            break

        else:
            (path,folder,filename,data,token) = message

            storage_dir = os.path.join(path,folder)

            if not os.path.exists(storage_dir):
                os.mkdir(storage_dir)

            file_savename = os.path.join(storage_dir,filename)
            print(storage_dir,filename)

            if file_savename.endswith('tif'):
                t0 = time.time()
                tif.imsave(file_savename,data)
                print('save time: ', time.time() - t0)

            else:
                t0 = time.time()
                io.imsave(file_savename,data)
                print('save time: ', time.time() - t0)

            message = None


