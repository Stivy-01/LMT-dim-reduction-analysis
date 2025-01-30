#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="lda",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    description="LMT Data Analysis Package",
    author="Andrea Stivala",
    install_requires=[
        'pandas',
        'numpy',
        'pathlib',
    ],
    python_requires='>=3.6',
) 