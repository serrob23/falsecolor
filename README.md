# falseColoring

Python module for H&E pseudo coloring for greyscale fluorescent images of datasets with nuclear and cytoplasmic staining.


## Installation

Run setup.py install while in the working directory.

```bash
python setup.py install
```


## Usage

```python
from FalseColor.FCdataobject import DataObject
import FalseColor.Color as fc
```
For CPU batch processing load data into DataObject:
(See Example/example notebook.ipynb)
```python
data_path = 'path/to/data' #contains .h5 file
dataSet = DataObject(data_path)

#zips data into imageSet property of Dataobject 
#imageSet will be a 4D array of images with [Z,X,Y,C]
Dataset.setupH5data() 
```
Batch process data using DataObjects processImages method:
```python
#method and kwargs are put into a dictionary
runnable_dict = {'runnable' : fc.falseColor, 'kwargs' : None}

#runnable_dict and desired images are passed into processImages method
pseudo_colored_data = Dataset.processImages(runnable_dict, Dataset.imageSet)

```

Several methods within Color.py are setup with GPU acceleration using numba.cuda.jit:
(See Example/GPU examples.ipynb)

```python
#set color levels for false coloring using background subtraction
nuclei_RGBsettings = [R,G,B] # list of floats (0.0:1.0) for color levels in nuclear channel
cyto_RGBsettings = [R,G,B] # list of floats (0.0:1.0) for color levels in cyto channel

#nuclei,cyto are 2D numpy arrays for false coloring see GPU example.ipynb for more details
pseudo_colored_data = fc.rapidFalseColor(nuclei,cyto,nuclei_RGBsettings,cyto_RGBsettings, run_normalization=False)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
