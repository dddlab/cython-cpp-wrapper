from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

import eigency

extensions = [
    Extension(
        "ccec.ccec",
        ["ccec/ccec.pyx"],
        include_dirs=[".", "ccec", "ccec/eigen-3.4.0"] + eigency.get_includes(include_eigen=False),
        language = "c++"
    ),
]

setup(
    name="ccec",
    version="0.0.0",
    ext_modules=cythonize(extensions),
    packages=["ccec"],
)
