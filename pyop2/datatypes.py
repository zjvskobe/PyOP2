from __future__ import absolute_import, print_function, division

import ctypes

import numpy
from petsc4py.PETSc import IntType, RealType, ScalarType

IntType = numpy.dtype(IntType)
RealType = numpy.dtype(RealType)
ScalarType = numpy.dtype(ScalarType)


def as_cstr(dtype):
    """Convert a numpy dtype like object to a C type as a string."""
    return {"bool": "unsigned char",
            "int": "int",
            "int8": "int8_t",
            "int16": "int16_t",
            "int32": "int32_t",
            "int64": "int64_t",
            "uint8": "uint8_t",
            "uint16": "uint16_t",
            "uint32": "uint32_t",
            "uint64": "uint64_t",
            "float32": "float",
            "float64": "double"}[numpy.dtype(dtype).name]


def as_ctypes(dtype):
    """Convert a numpy dtype like object to a ctypes type."""
    return {"bool": ctypes.c_bool,
            "int": ctypes.c_int,
            "int8": ctypes.c_char,
            "int16": ctypes.c_int16,
            "int32": ctypes.c_int32,
            "int64": ctypes.c_int64,
            "uint8": ctypes.c_ubyte,
            "uint16": ctypes.c_uint16,
            "uint32": ctypes.c_uint32,
            "uint64": ctypes.c_uint64,
            "float32": ctypes.c_float,
            "float64": ctypes.c_double}[numpy.dtype(dtype).name]


class _MapMask(ctypes.Structure):
    _fields_ = [("section", ctypes.c_voidp),
                ("indices", ctypes.c_voidp)]


class _EntityMask(ctypes.Structure):
    _fields_ = [("section", ctypes.c_voidp),
                ("bottom", ctypes.c_voidp),
                ("top", ctypes.c_voidp)]
