from setuptools import setup

with open('requirements.txt','r') as f:
    requires = f.read().splitlines()
f.close()

setup(name='falsecolor',
    packages = ['falsecolor'],
    version='1.1.3.9',
    license = 'GNU Affero General Public License v3.0',
    description='Methods for H&E pseudo coloring of grayscale fluorescent images',
    author='Robert Serafin',
    author_email='serrob23@uw.edu',
    url = 'https://github.com/serrob23/FalseColor',
    download_url = 'https://github.com/serrob23/FalseColor/releases/download/v1.1.3.8/falsecolor-1.1.3.8.tar.gz',
    keywords = ['Image Analysis', 
                            'Fluorescence Microscopy', 
                            'Color Translation', 
                            'Virtual Staining',
                            'Histology'],
    install_requires = requires,
    python_requires='>=3.6',

)
