#!/usr/bin/env python3
# coding: utf-8

from setuptools import setup

setup(
    name="Machine Learning",
    version="0.1",
    license="MIT",
    python_requires="==3.9.12",
    zip_safe=False,
    include_package_data=True,
    packages=["WarmUp", "normalEq", "PlotData", "ComputeCost", "FeatureScaling"],
    package_dir={
        "normalEq": ".",
        "WarmUp": "./Warmup",
        "PlotData": "./PlotData",
        "ComputeCost": "./ComputeCost",
        "FeatureScaling": "./FeatureScaling",
    }
)
