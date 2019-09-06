"""
Script for rapid false coloring large datasets

Rob Serafin
08/28/2019
"""

import os
import glob
import FalseColor_methods as fc
import numpy 
import cv2
from FCdataobject import DataObject

falseColor_runnable = {'runnable':fc.falseColor,'kwargs':{'channelIDs':['s00','s01']}}
