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
For CPU batch processing load data into DataObject, currently setup to work with .h5 files:
(See Example/example notebook.ipynb)
```python
data_path = 'path/to/data' #contains .h5 file
dataSet = DataObject(data_path)

#zips data into imageSet property of Dataobject 
#imageSet will be a 4D array of images with [Z,X,Y,C]
Dataset.setupH5data() 
