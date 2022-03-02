"""
Example data sets for digirock
"""
import os
import pooch
from . import __version__

GOODBOY = pooch.create(
    path=os.curdir,
    base_url="https://github.com/trhallam/digirock/raw/{version}/tests/test_data/",
    # Always get the main branch if dev in version. Thick package doesn't use dev releases.
    version=__version__ + "+dirty" if "dev" in __version__ else __version__,
    # If this is a development version, get the data from the master branch
    version_dev="main",
    # The registry specifies the files that can be fetched from the local storage
    registry={
        "COMPLEX_PVT.inc": "3018c7ec33dded551e0bcd44103a1abd27ff4895268c712197616e396532da25",
        "PVT_BO.inc": "053669c122948b690b03bcd2e5d11bdbc377bf84cddcd0d614ee19ec22ca36b6",
        "PVT_RS.inc": "ff869731b2ece69fa0686b6a0204f113a0106e359413ddf1547841cbdf3d219d",
    },
)


def fetch_example_data():
    return {key: GOODBOY.fetch(key) for key in GOODBOY.registry}
