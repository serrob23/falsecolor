from distutils.core import setup

setup(name='FalseColor',
    version='0.1.1',
    description='Methods for H&E pseudo coloring of grayscale images',
    author='Robert Serafin',
    install_requires =[
                        'opencv-python',
                        'numpy',
                        'scikit-image',
                        'scipy',
                        'pathos',
                        'numba',
                        'tifffile',
                        'h5py',
                        'functools',
                        'pathos',
                        ],
    url = 'https://github.com/serrob23/falseColoring', 
    author_email='serrob23@uw.edu',
    packages=['FalseColor'])