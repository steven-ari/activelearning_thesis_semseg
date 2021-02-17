#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from os import path
from setuptools import setup, find_packages


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
history = ""

requirements = [
                'setuptools~=46.1.3',
                'pip',
                'torch~=1.5.0',
                'torchvision~=0.6.0',
                'numpy~=1.18.2',
                'xgboost~=1.0.2',
                'Pillow~=7.1.1',
                'matplotlib~=3.2.1',
                'kmeans-pytorch~=0.3',
                'scikit-learn~=0.23.1',
                'seaborn~=0.10.0',
                'tqdm~=4.46.1',
    ]

setup_requirements = []

test_requirements = []

setup(
    author="Steven Tjong",
    author_email='stevenari94@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
    description="Active Learning prototype",
    install_requires=requirements,
    long_description=long_description + '\n\n' + history,
    include_package_data=True,
    name='active_learning_prototypes',
    packages=find_packages(),
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    test_suite='test',
    url='https://github.pie.apple.com/turi-tag/active-learning-prototypes/',
    entry_points={
    },
    version='0.1.0',
    zip_safe=False,
)