from setuptools import setup

def readme():
	with open('README.md') as f:
		return f.read()

setup(
	name='distanceclosure',
	version='0.1',
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
	packages=['distanceclosure'],
	install_requires=[
		'numpy',
		'scipy',
	],
	include_package_data=True,
	zip_safe=False,
	)