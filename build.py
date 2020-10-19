#!/usr/bin/python

import py_compile

py_compile.compile("./check_face.py",cfile="check_face.pyo",optimize=2)
py_compile.compile("./learn_face.py",cfile="learn_face.pyo",optimize=2)
py_compile.compile("./find_face.py",cfile="find_face.pyo",optimize=2)
