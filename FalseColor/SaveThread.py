"""
Save Thread for saving large datasets to tif format

Rob Serafin
9/12/2019
"""
import os
import tifffile as tif
import numpy
import cv2


def saveProcess(queue):
    """
    queue : multiprocessing queue

        message : result of queue.get()
            has the following properties in this order:

                path : str or pathlike
                    top level storage directory for data
                
                folder : str
                    specific dir to save data in

                filename : str
                    "file.tif" filename for data

                data : numpy array
                    image to save

                token : None or str
                    token will be a str when thread stop is called
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

            if file_savename.endswith('tif'):
                tif.imsave(file_savename,data)

            else:
                cv2.imwrite(data,file_savename)


