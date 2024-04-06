#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===============================
HtmlTestRunner
===============================


.. image:: https://img.shields.io/pypi/v/scm_irl.svg
        :target: https://pypi.python.org/pypi/scm_irl
.. image:: https://img.shields.io/travis/esquivelrs/scm_irl.svg
        :target: https://travis-ci.org/esquivelrs/scm_irl

Inverse rl applied to navigation of marine autonomous robots


Links:
---------
* `Github <https://github.com/esquivelrs/scm_irl>`_
"""

from setuptools import setup, find_packages

requirements = ['Click>=6.0', 'numpy', 'gymnasium', 'matplotlib', 'scipy', 'pandas', 'seaborn', 'pygame', 'shapely', 'rasterio', 'opencv-python']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Rolando Esquivel Sancho",
    author_email='rolando.esq@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="Inverse rl applied to navigation of marine autonomous robots",
    entry_points={
        'console_scripts': [
            'scm_irl=scm_irl.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=__doc__,
    include_package_data=True,
    keywords='scm_irl',
    name='scm_irl',
    packages=find_packages(include=['scm_irl', 'scm_irl.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/esquivelrs/scm_irl',
    version='0.1.0',
    zip_safe=False,
)
