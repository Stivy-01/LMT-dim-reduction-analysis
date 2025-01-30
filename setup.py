#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="lmt_analysis",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    description="LMT Dimensionality Reduction Analysis Toolkit",
    author="Andrea Stivala",
    author_email="andreastivala.as@gmail.com",
    install_requires=[
        'numpy==1.23.5',
        'scipy>=1.9.0',
        'pandas',
        'scikit-learn>=1.0.0',
        'plotly>=5.13.0',
        'tkcalendar>=1.6.1',
        'seaborn>=0.12.2',
        'pathlib',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
) 