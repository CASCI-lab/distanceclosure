from setuptools import setup, find_packages
from setuptools.extension import Extension
from distanceclosure import __package__, __description__, __version__

#from Cython.Build import cythonize
#from Cython.Distutils import build_ext

#import subprocess


# Readme
def readme():
    with open('README.md') as f:
        return f.read()


# Remove old files
#subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
#subprocess.Popen("rm -rf distanceclosure/cython/*.c", shell=True, executable="/bin/bash")
#subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")
#
#
#
#ext_modules = [
#    Extension("distanceclosure.cython.dijkstra", sources=[
#        'distanceclosure/cython/pqueue.pxd',
#        'distanceclosure/cython/dijkstra.pyx',
#        'distanceclosure/cython/libpqueue/pqueue.c',
#    ]),
#]
#cmdclass = {'build_ext': build_ext}

setup(
    name=__package__,
    version=__version__,
    description=__description__,
    long_description="Methods to calculate distance closure on complex networks as defined in T. Simas and L.M. Rocha [2015].\"Distance Closures on Complex Networks\". Network Science, 3(2):227-268. doi:10.1017/nws.2015.11",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords="networks distance closure graph",
    url="http://github.com/rionbr/distanceclosure",
    author="Rion Brattig Correia",
    author_email="rionbr@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={

    },
    install_requires=[
        'numpy',
        'scipy',
        'networkx',
        'pandas',
        #'cython',  # for Dijkstra
    ],
    #cmdclass=cmdclass,
    include_package_data=True,
    zip_safe=False,
    #ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)
