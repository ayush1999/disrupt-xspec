import os
#from distutils.core import setup, Extension
from setuptools import setup, find_packages, Extension
import numpy.distutils.misc_util
import imp 

version = imp.load_source('xsmodels.version', 'version.py')

XSDIR = os.environ["HEADAS"]

inc = numpy.distutils.misc_util.get_numpy_include_dirs()
inc.append(XSDIR+"/include")
libs = ["XSFunctions", "XSModel", "XSUtil", "XS"]
libdirs = [XSDIR+"/lib"]

setup(
    name='xsmodels',
    version=version.version,
    description='XSModels',
    ext_modules=[Extension("_xsmodels", ["xsmodels/_xsmodels.c"],
    include_dirs=inc,
    libraries=libs,
    library_dirs=libdirs)],
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    license='GPL',
    install_requires=[
        'numpy>=1.10'
    ],
    extras_require={
        'docs': ['numpydoc']
    }


)
