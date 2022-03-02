"""
Example data sets for digirock
"""
import os
import pooch
from . import __local_version__

GOODBOY = pooch.create(
    path=os.curdir,
    base_url="https://raw.githubusercontent.com/trhallam/digirock/{version}/tests/test_data/",
    version=__local_version__,
    # If this is a development version, get the data from the master branch
    version_dev="main",
    # The registry specifies the files that can be fetched from the local storage
    registry={
        "COMPLEX_PVT.inc": "md5:492cb074b8232283b02e18a18a7777ec",
        "PVT_BO.inc": "md5:1582ab00bc170cecd21bdfd5d3221f81",
        "PVT_RS.inc": "md5:aa882bc90020b529d73d346767e46481",
    },
)


def fetch_example_data():
    return {key: GOODBOY.fetch(key) for key in GOODBOY.registry}
