#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

# System modules


# External modules

# Internal modules

from setuptools import find_packages, setup

setup(
    name='ocean_flow',
    packages=find_packages(
        include=['ocean_flow']
    ),
    version='0.1',
    description='Ocean surface reconstruction with Bayesian flow matching',
    author='Chiara Zelco',
    license='MIT',
)
