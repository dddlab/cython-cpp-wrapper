import os, pathlib
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from subprocess import check_call

import eigency

HERE = pathlib.Path(__file__).parent

extensions = [
    Extension(
        "ccec.ccec",
        ["ccec/ccec.pyx"],
        include_dirs=[".", "ccec", "ccec/eigen"] + eigency.get_includes(include_eigen=False),
        language = "c++"
    ),
]

def gitcmd_update_submodules():
	'''	Check if the package is being deployed as a git repository. If so, recursively
		update all dependencies.

		@returns True if the package is a git repository and the modules were updated.
			False otherwise.
	'''
	if os.path.exists(os.path.join(HERE, '.git')):
		check_call(['git', 'submodule', 'update', '--init', '--recursive'])
		return True

	return False

class gitcmd_develop(develop):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure.
	'''
	def run(self):
		gitcmd_update_submodules()
		develop.run(self)

class gitcmd_install(install):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure.
	'''
	def run(self):
		gitcmd_update_submodules()
		install.run(self)

class gitcmd_sdist(sdist):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure;.
	'''
	def run(self):
		gitcmd_update_submodules()
		sdist.run(self)


setup(
    cmdclass={
		'develop': gitcmd_develop, 
		'install': gitcmd_install, 
		'sdist': gitcmd_sdist,
	}, 
    name="ccec",
    version="0.0.0",
    ext_modules=cythonize(extensions),
    packages=["ccec"],
)
