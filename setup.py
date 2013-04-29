from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='dpcluster',
      version='0.1',
      description='Dirichlet Process clustering',
      long_description=readme(),
      url='http://github.com/moldovan/dpcluster',
      author='Teodor Mihai Moldovan',
      author_email='moldovan@cs.berkeley.edu',
      license='MIT',
      packages=['dpcluster'],
      install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
        ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

