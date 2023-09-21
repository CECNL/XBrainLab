#!/usr/bin/env python
import setuptools
from os import path

def parse_requires():
    requirements = list()
    with open('requirements.txt', "r") as fh:
        for line in fh:
            req = line.strip()
            if req.startswith("#"):
                continue
            req = req.split("#", maxsplit=1)[0].strip()
            requirements.append(req)
    return requirements

NAME = "XBrainLab"
DESCRIPTION = (
    "XBrainLab is a EEG decoding toolbox with deep learning "
    "dedicated to neuroscience discoveries."
)
AUTHOR = "CECNL"
AUTHOR_EMAIL = "cecnl@nctu.edu.tw"
LICENSE = "GNU General Public License v3.0"
URL = "https://github.com/CECNL/XBrainLab"
KEYWORDS = "neuroscience deep-learning brain-state-decoding EEG"
if __name__ == "__main__":

    with open(path.join("README.md"), "r", encoding="utf-8") as fh:
        long_description = fh.read()

    install_requires = parse_requires()

    setuptools.setup(
        name=NAME,
        author=AUTHOR, 
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        entry_points={
            'console_scripts': [
                'XBrainLab=XBrainLab.ui:main'
            ]
        },
        url=URL,
        install_requires=install_requires,
        packages=setuptools.find_packages(),
        use_scm_version=True,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: Python",
            'Topic :: Software Development',
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
        platforms="any",
        python_requires=">=3.9",
        keywords=KEYWORDS,
        zip_safe=False,
    )