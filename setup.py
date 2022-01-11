#from distutils.core import setup

from setuptools import Extension, setup
from Cython.Build import cythonize
from numpy import get_include

ext_modules = [
    Extension(
        "chansim.utils.fastCalcA",
        ["chansim/utils/fastCalcA.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "chansim.utils._fastInterp",
        ["chansim/utils/_fastInterp.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]


setup(name='chansim',
      version='0.1',
      author='Matt Covington, Max Cooper',
      packages=['chansim','chansim.utils', 'chansim.viz'],
      include_dirs =[get_include()],
      ext_modules = cythonize(ext_modules,
              compiler_directives={'language_level':3,}
          ),
      zip_safe=False,
      )
