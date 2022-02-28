# -*- coding: utf8 -*-

"""File utilities

    Loading data from custom file types, e.g. Eclipse Data Files

"""

import itertools
import string
import pandas as pd
import numpy as np

from collections import OrderedDict

from ._utils import ndim_index_list

ECLIPSE_typemap = {"INTE": int, "REAL": float, "CHAR": str, "LOGI": bool, "DOUB": float}


def get_type(a):
    """returns the type of a string

    Args:
        a ([type]): [description]
    """
    f, i, ch = False, False, False

    try:
        float(a)
    except ValueError:
        pass
    else:
        f = True

    try:
        int(a)
    except ValueError:
        pass
    else:
        i = True

    try:
        str(a)
    except ValueError:
        pass
    else:
        ch = True

    if i:
        return ECLIPSE_typemap["INTE"]
    elif f:
        return ECLIPSE_typemap["REAL"]
    elif ch:
        return ECLIPSE_typemap["CHAR"]
    else:
        print("Unknown type: assuming string")
        return str


def scan_eclipsekw(filepath, overloaded_kw=False):
    """Scans an eclipse row-wise property file from ascii to determine which
    keywords are present.

    This is for Eclipse key word formatted files such as the .PETREL files
    exported from ETLPY-M and files exported from Petrel simulation models in
    X format.

    Args:
        filepath (str): The fill file path and name.

    Returns:
        keywords (OrderedDict): A list of keyword strings identified in the file.
    """
    efile = open(filepath, mode="r")
    print(f"Scanning {filepath}")
    sections = OrderedDict()
    known_sections_count = dict()
    new_section = True

    for i, line in enumerate(efile):
        if "--" in line[0:2] or len(line) < 2:
            pass  # skip comment lines
        elif "/" in line:
            new_section = True
            cursec["last_line"] = i
            sections[keyw] = cursec
        elif new_section:
            cursec = dict()
            # print('new', line)
            keyw = line.split("--")[0]
            keyw = keyw.rstrip("\n").rstrip()
            if overloaded_kw:
                if keyw in known_sections_count.keys():
                    known_sections_count[keyw] = known_sections_count[keyw] + 1
                else:
                    known_sections_count[keyw] = 0
                keyw = f"{keyw}_{known_sections_count[keyw]}"
            cursec["first_line"] = i
            new_section = False
        else:
            pass  # skip data lines
    efile.close()
    return sections


def read_eclipsekw(filepath, filter=None, null=-9999, ijk=None, overloaded_kw=False):
    """Read an eclipse row-wise property file from ascii into a DataFrame

    This is for Eclipse key word formatted files such as the .PETREL files
    exported from ETLPY-M and files exported from Petrel simulation models in
    X format.

    Args:
        filepath (str): The full file path and name.
        filter ([type], optional): Defaults to None. A list of key words to load
            if none all keywords are loaded.

    Returns:
        DataFrame: A pandas DataFrame with columns corresponding to key words.
    """
    sections = scan_eclipsekw(filepath, overloaded_kw=overloaded_kw)
    df = pd.DataFrame()
    if filter is None:
        load_filter = sections.keys()
    else:
        load_filter = list()
        for key in filter:
            if key not in sections.keys():
                print(f"{key} section is missing from {filepath}")
            else:
                load_filter.append(key)

    if isinstance(ijk, list) and len(ijk) == 3:
        ijk.reverse()
        index = ndim_index_list(ijk)
        df["i"] = index[2]
        df["j"] = index[1]
        df["k"] = index[0]

    for key in load_filter:
        sec = sections[key]
        sfl = sec["first_line"]
        sll = sec["last_line"]
        print("Loading section data for: ", key, " Lines: ", sfl + 1, " to ", sll)
        efile = open(filepath, mode="r")
        tdata = list()
        has_data = sll > sfl
        # number type properties
        if has_data:
            for i, line in enumerate(efile):
                if i <= sfl or "--" in line:
                    pass
                elif i <= sll:
                    cache_line = line.split()
                    for (
                        dp
                    ) in (
                        cache_line
                    ):  # check for crazy Eclipse data multipliers e.g. 10*dp
                        if "*" in dp:
                            n, d = dp.split("*")
                            tdata.extend([d] * int(n))
                        else:
                            tdata.append(dp)
                else:
                    pass

            # petrel GRDECL and ECLIPSE input check
            if tdata[-1] == "/":
                popped = tdata.pop(-1)
            df[key] = tdata
            # try int then float
            try:
                df[key] = df[key].astype(int)
                df[key].replace(null, np.nan, inplace=True)
            except (ValueError, OverflowError):
                try:
                    df[key] = df[key].astype("float")
                    df[key].replace(float(null), np.nan, inplace=True)
                except ValueError:
                    pass
    return df


def _read_eclipsekw_section(filepath, kword):
    """Helper function for read_eclipsekw_*****"""
    found_kw = False
    table = ""
    with open(filepath, "r") as f:
        for line in f:
            if line[:2] == "--":
                pass  # comments
            elif kword in line:
                found_kw = True
            elif found_kw and line[0] not in string.ascii_uppercase:
                line = line[::-1]
                line = line[line.rfind("--") + 2 :].strip()  # remove trailling comments
                table = table + "  " + line[::-1]
            elif found_kw and line[0] in string.ascii_uppercase:
                break  # end of section
            else:
                pass  # blank lines probs
    return table


def read_eclipsekw_2dtable(filepath, kword):
    """A 2D table has the same number of items per row and will return a
    contiguous list of table row items.

    Comments on lines are denoted by '--' in the first two space.
    Blank lines are ignored.
    Sections are terminated by an uppercase ascii character in column 0.

    Example Input:

    DENSITY
    -- OilDens   WaterDens    GasDens
    -- kg/m3       kg/m3       kg/m3
    882.0      1101.3      1.09956 / -- #1
    882.0      1101.3      1.09956 /

    Args:
        kword (str): The keyword to search the input file for.

    Raises:
        ValueError: [description]
    """
    table = _read_eclipsekw_section(filepath, kword)
    table = table.strip().split("/")[:-1]
    table = [p.strip().split() for p in table]
    return table


def read_eclipsekw_3dtable(filepath, kword):
    """

    Args:
        filepath ([type]): [description]
        kword ([type]): [description]

    Raises:
        ValueError: [description]
    """
    table = _read_eclipsekw_section(filepath, kword)
    table = table.split("/  /")[:-1]
    table = [p.strip().split("/") for p in table]
    return table


def read_eclipsekw_m3dtable(filepath, kword):
    """

    Args:
        filepath ([type]): [description]
        kword ([type]): [description]

    Raises:
        ValueError: [description]
    """
    raise NotImplementedError
