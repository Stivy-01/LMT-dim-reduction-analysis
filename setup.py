#!/usr/bin/env python3
"""
LMT Toolkit - Live Mouse Tracker thesis analysis pipeline.
"""
from setuptools import setup, find_packages

setup(
    name="lmt-toolkit",
    version="1.0.0",
    author="Andrea Stivala",
    author_email="andreastivala.as@gmail.com",
    description="Thesis analysis pipeline and figure builders for LMT behavioral data",
    long_description=open("docs/readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Stivy-01/LMT-dim-reduction-analysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "openpyxl>=3.1.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "statsmodels>=0.14.0",
        "Pillow>=9.0.0",
        "tkcalendar>=1.6.1",
    ],
    extras_require={
        "riemannian": ["pyriemann>=0.3.0"],
        "dev": ["pytest>=7.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="behavior analysis, mouse tracking, dimensionality reduction, LDA, PCA",
    entry_points={
        "console_scripts": [
            "thesis-analysis=src.analysis.thesis_analysis:main",
            "lmt-build-all=src.visualization.build_all:main",
        ],
    },
)
