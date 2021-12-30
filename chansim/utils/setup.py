from setuptools import Extension, setup
from Cython.Build import cythonize
from numpy import get_include

ext_modules = [
    Extension(
        "fastCalcA",
        ["fastCalcA.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(include_dirs =[get_include],
    ext_modules = cythonize(ext_modules,
        compiler_directives={'language_level':3,}
    ),

)
