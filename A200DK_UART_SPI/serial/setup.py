#!/usr/bin/env python

from setuptools import setup, Extension


setup(
    name		= "a200dkserial",
	version		= "0.0.1",
    packages    = ['src'],
	description	= "serial operation for A200DK",
    ext_modules	= [Extension("a200dkserial", ["src/c_serial.c", "src/py_serial.c"])]
)
