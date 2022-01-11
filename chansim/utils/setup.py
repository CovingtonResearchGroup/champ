from setuptools import Extension, setup
from Cython.Build import cythonize
from numpy import get_include

ext_modules = [
    Extension(
        "fastCalcA",
        ["fastCalcA.pyx"]
    ),
    Extension(
        "_fastInterp",
        ["_fastInterp.pyx"]
    ),
]

setup(include_dirs =[get_include],
    ext_modules = cythonize(ext_modules,
        compiler_directives={'language_level':3,}
    ),

)
