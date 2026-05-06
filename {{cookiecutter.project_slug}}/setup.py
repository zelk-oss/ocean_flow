#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules


# External modules

# Internal modules

from setuptools import find_packages, setup

setup(
    name='{{cookiecutter.project_slug}}',
    packages=find_packages(
        include=['{{cookiecutter.project_slug}}']
    ),
    version='{{cookiecutter.start_version}}',
    description='{{cookiecutter.description}}',
    author='{{cookiecutter.author_name}}',
    license='MIT',
)
