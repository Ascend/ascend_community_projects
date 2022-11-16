#!/usr/bin/env python

from setuptools import setup, Extension


setup(
    name		= "spidev",
    version		= "0.0.1",
    description	= "serial operation for A200DK",
    ext_modules	= [Extension("spidev", ["src/c_spidev.c","src/py_spidev.c"])]
)
