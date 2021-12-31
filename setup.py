#from distutils.core import setup

from setuptools import Extension, setup
from Cython.Build import cythonize
from numpy import get_include

ext_modules = [
    Extension(
        "fastCalcA",
<<<<<<< HEAD
        ["chansim/utils/fastCalcA.pyx"]
=======
        ["chansim/utils/fastCalcA.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
>>>>>>> 00d67f6938c247ea440e663a41eaf8ccb2269cd6
    )
]


setup(name='chansim',
      version='0.1',
      author='Matt Covington, Max Cooper',
      packages=['chansim','chansim.utils', 'chansim.viz'],
      include_dirs =[get_include()],
      ext_modules = cythonize(ext_modules,
              compiler_directives={'language_level':3,}
          ),
      )
