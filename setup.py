# from distutils.core import setup

from setuptools import Extension, setup
from Cython.Build import cythonize
from numpy import get_include

ext_modules = [
    Extension(
        "champ.utils.*",
        ["champ/utils/*.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]


setup(
    name="champ",
    version="0.2",
    author="Matt Covington, Max Cooper",
    packages=["champ", "champ.utils", "champ.viz"],
    include_dirs=[get_include()],
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": 3,}),
    zip_safe=False,
    entry_points={"console_scripts": ["champ-runSim=champ.runSim:main"],},
    install_requires=['numpy', 
                      'scipy', 
                      'pyyaml', 
                      'cython',
                      'pillow', 
                      'matplotlib',
                      'setuptools',
                      'mayavi',
                      ],
)
