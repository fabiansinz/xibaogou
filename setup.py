#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

long_description = "Spike triggered mixture model (STM) based learning algorithm to detect cells in stacks."


setup(
    name='xibaogou',
    version='0.1.0.dev1',
    description="Cell detection algorithm.",
    long_description=long_description,
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    license="MIT License",
    url='https://github.com/fabiansinz/xibaogou',
    keywords='machine learning, computational biology',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy','theano'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT ',
        'Topic :: Database :: Front-Ends',
    ],
)
