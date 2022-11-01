from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(
      name='nms_module',
      ext_modules=cythonize('cython_nms.pyx'),
)

