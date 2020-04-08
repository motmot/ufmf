from setuptools import setup, find_packages
from os import path
from io import open

# read the contents of README.md
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="motmot.ufmf",
    version="0.4.2",
    description="micro-fmf (.ufmf) format library",
    author="Andrew Straw",
    author_email="strawman@astraw.com",
    namespace_packages=["motmot"],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ufmf2fmf = motmot.ufmf.ufmf2fmf:main",
            "ufmfcat = motmot.ufmf.ufmfcat:main",
            "fmf2ufmf = motmot.ufmf.fmf2ufmf:main",
            "ufmfstats = motmot.ufmf.ufmfstats:main",
            "ufmf_reindex = motmot.ufmf.reindex:main",
            "ufmf_1to3 = motmot.ufmf.ufmf_1to3:main",
            "ufmf_2to3 = motmot.ufmf.ufmf_2to3:main",
        ],
        "motmot.fview.plugins": "fview_ufmf_saver = motmot.ufmf.ufmf_flytrax:Tracker",
    },
    package_data={"motmot.ufmf": ["*.xrc",]},
    test_suite="nose.collector",
)
