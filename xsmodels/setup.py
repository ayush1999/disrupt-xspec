import os
from distutils.core import setup, Extension
import numpy.distutils.misc_util

XSDIR = os.environ["HEADAS"]

inc = numpy.distutils.misc_util.get_numpy_include_dirs()
inc.append(XSDIR+"/include")
libs = ["XSFunctions", "XSModel", "XSUtil", "XS"]
libdirs = [XSDIR+"/lib"]

setup(
    ext_modules=[Extension("_xsmodels", ["_xsmodels.c"],
    include_dirs=inc,
    libraries=libs,
    library_dirs=libdirs)]
)
