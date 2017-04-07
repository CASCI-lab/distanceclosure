from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

try:
	from Cython.Distutils import build_ext
except:
	USE_CYTHON = False
else:
	USE_CYTHON = True
from distanceclosure import __version__
import subprocess

# Readme
def readme():
	with open('README.md') as f:
		return f.read()

# Remove old files
subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
subprocess.Popen("rm -rf distanceclosure/cython/*.c", shell=True, executable="/bin/bash")
subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")
#
#
#
cmdclass = {}
ext_modules = []
#
if USE_CYTHON:
	ext_modules += [
		Extension("distanceclosure.cython._dijkstra", sources=['distanceclosure/cython/_dijkstra.pyx', 'distanceclosure/cython/libpqueue/pqueue.c'])
	]
	cmdclass.update({'build_ext':build_ext})
else:
	ext_modules += [
		Extension("distanceclosure.cython._dijkstra", sources=['distanceclosure/cython/_dijkstra.c', 'distanceclosure/cython/libpqueue/pqueue.c'])
	]

setup(
	name='distanceclosure',
	version=__version__,
	description="Distance Closure on Complex Networks",
	long_description="Methods to calculate distance closure on complex networks as defined in T. Simas and L.M. Rocha [2015].\"Distance Closures on Complex Networks\". Network Science, 3(2):227-268. doi:10.1017/nws.2015.11",
	classifiers=[
		'Development Status :: 3 - Alpha',
		'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
		'Programming Language :: Python :: 2.7',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Information Analysis',
	],
	keywords="networks distance closure graph",
	url="http://github.com/rionbr/distanceclosure",
	author="Rion Brattig Correia",
	author_email="rionbr@gmail.com",
	license="GPL 2.0",
	packages=find_packages(),
	install_requires=[
		'numpy',
		'scipy',
		'joblib', # for Parallel 
		'cython', # for Dijkstra
	],
	ext_modules=cythonize(ext_modules),
	cmdclass=cmdclass,
	include_package_data=True,
	package_data={

	},
	zip_safe=False,
	)

