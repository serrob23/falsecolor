from distutils.core import setup

setup(name='FalseColor',
    packages = ['FalseColor'],
    version='1.0.0',
    license = 'GNU Affero General Public License v3.0',
    description='Methods for H&E pseudo coloring of grayscale fluorescent images',
    author='Robert Serafin',
    author_email='serrob23@uw.edu',
    url = 'https://github.com/serrob23/FalseColor',
    download_url = '',
    keywords = ['Image Analysis', 
                            'Fluorescence Microscopy', 
                            'Color Translation', 
                            'Virtual Staining',
                            'Histology'],
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
                        'matplotlib',
                        'json',
                        'os'
                        ],
    classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: GNU Affero General Public License v3.0'
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
    ],

                        )