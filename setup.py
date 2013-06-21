from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='dpcluster',
      version='0.104',
      description='dpcluster is a package for grouping together (clustering) vectors. It automatically chooses the number of clusters that fits the data best based on the underlying Dirichlet Process mixture model.',
      long_description=readme(),
      classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Development Status :: 2 - Pre-Alpha'
        ],
      url='http://dpcluster.readthedocs.org/',
      download_url='http://github.com/teodor-moldovan/dpcluster',
      author='Teodor Mihai Moldovan',
      author_email='moldovan@cs.berkeley.edu',
      packages=['dpcluster'],
      install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
        ],
      # py_modules=['dpcluster'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

