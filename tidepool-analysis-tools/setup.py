#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description: setup script for the tidals package
created: 2018-10-01
author: Ed Nykaza
license: BSD-2-Clause
Parts of this file were taken from
https://packaging.python.org/tutorials/packaging-projects/
"""

# %% REQUIRED LIBRARIES
import setuptools
import os
import glob
import shutil

# %% remove the excess files that were downloaded from github
# TODO: make tidals its own repository under tidepool_org
# once tidals becomes its own github repository, then this step will no longer be necessary
# TODO: publish tidals as a PyPi pacakge

scrFiles = glob.glob(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "*")))

for i in scrFiles:
    if "tidepool-analysis-tools" not in i:
        if os.path.isdir(i):
            shutil.rmtree(i)
        else:
            os.remove(i)

# %% START OF SETUP
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tidals",
    version="0.0.1",
    author="Ed Nykaza",
    author_email="ed@tidepool.org",
    description="Tidepool Data Analysis Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tidepool-org/data-analytics/tree/master/tidal-analysis-tools/tidals",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-2-Clause",
        "Operating System :: OS Independent",
    ],
)


