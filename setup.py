import setuptools
import numpy as np
from setuptools.extension import Extension
from Cython.Build import cythonize
from pathlib import Path

extensions = [
    Extension(
        "pybasicbayes.util.cstats",
        ["pybasicbayes/util/cstats.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-w"],
    )
]


setuptools.setup(
    ext_modules=cythonize(extensions, compiler_directives={"boundscheck": False, "wraparound": False, "cdivision": True, 'language_level': 3}),
)
