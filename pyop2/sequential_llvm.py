"""OP2 sequential backend, generating LLVM IR"""

from exceptions import *
from petsc_base import *
import host

# Parallel loop API


class JITModule(host.JITModule):

# TODO Generate target datalayout, target triple depending on architecture.
    self._header = """
; ModuleID = '<somenamehere>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

; Type declarations
%struct._object = type { i32, %struct._typeobject* }
%struct._typeobject = type { i32, %struct._typeobject*, i32, i8*, i32, i32, void (%struct._object*)*, i32 (%struct._object*, %struct._IO_FILE*, i32)*, %struct._object* (%struct._object*, i8*)*, i32 (%struct._object*, i8*, %struct._object*)*, i32 (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*)*, %struct.PyNumberMethods*, %struct.PySequenceMethods*, %struct.PyMappingMethods*, i32 (%struct._object*)*, %struct._object* (%struct._object*, %struct._object*, %struct._object*)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, i32 (%struct._object*, %struct._object*, %struct._object*)*, %struct.PyBufferProcs*, i32, i8*, i32 (%struct._object*, i32 (%struct._object*, i8*)*, i8*)*, i32 (%struct._object*)*, %struct._object* (%struct._object*, %struct._object*, i32)*, i32, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*)*, %struct.PyMethodDef*, %struct.PyMemberDef*, %struct.PyGetSetDef*, %struct._typeobject*, %struct._object*, %struct._object* (%struct._object*, %struct._object*, %struct._object*)*, i32 (%struct._object*, %struct._object*, %struct._object*)*, i32, i32 (%struct._object*, %struct._object*, %struct._object*)*, %struct._object* (%struct._typeobject*, i32)*, %struct._object* (%struct._typeobject*, %struct._object*, %struct._object*)*, void (i8*)*, i32 (%struct._object*)*, %struct._object*, %struct._object*, %struct._object*, %struct._object*, %struct._object*, void (%struct._object*)*, i32 }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.PyNumberMethods = type { %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*, %struct._object*)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*)*, i32 (%struct._object*)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, i32 (%struct._object**, %struct._object**)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*)* }
%struct.PySequenceMethods = type { i32 (%struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, i32)*, %struct._object* (%struct._object*, i32)*, %struct._object* (%struct._object*, i32, i32)*, i32 (%struct._object*, i32, %struct._object*)*, i32 (%struct._object*, i32, i32, %struct._object*)*, i32 (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, %struct._object* (%struct._object*, i32)* }
%struct.PyMappingMethods = type { i32 (%struct._object*)*, %struct._object* (%struct._object*, %struct._object*)*, i32 (%struct._object*, %struct._object*, %struct._object*)* }
%struct.PyBufferProcs = type { i32 (%struct._object*, i32, i8**)*, i32 (%struct._object*, i32, i8**)*, i32 (%struct._object*, i32*)*, i32 (%struct._object*, i32, i8**)*, i32 (%struct._object*, %struct.bufferinfo*, i32)*, void (%struct._object*, %struct.bufferinfo*)* }
%struct.bufferinfo = type { i8*, %struct._object*, i32, i32, i32, i32, i8*, i32*, i32*, i32*, [2 x i32], i8* }
%struct.PyMethodDef = type { i8*, %struct._object* (%struct._object*, %struct._object*)*, i32, i8* }
%struct.PyMemberDef = type opaque
%struct.PyGetSetDef = type { i8*, %struct._object* (%struct._object*, i8*)*, i32 (%struct._object*, %struct._object*, i8*)*, i8*, i8* }
%struct.PyArrayObject = type { i32, %struct._typeobject*, i8*, i32, i32*, i32*, %struct._object*, %struct._PyArray_Descr*, i32, %struct._object* }
%struct._PyArray_Descr = type { i32, %struct._typeobject*, %struct._typeobject*, i8, i8, i8, i8, i32, i32, i32, %struct._arr_descr*, %struct._object*, %struct._object*, %struct.PyArray_ArrFuncs*, %struct._object* }
%struct._arr_descr = type { %struct._PyArray_Descr*, %struct._object* }
%struct.PyArray_ArrFuncs = type { [21 x void (i8*, i8*, i32, i8*, i8*)*], %struct._object* (i8*, i8*)*, i32 (%struct._object*, i8*, i8*)*, void (i8*, i32, i8*, i32, i32, i32, i8*)*, void (i8*, i8*, i32, i8*)*, i32 (i8*, i8*, i8*)*, i32 (i8*, i32, i32*, i8*)*, void (i8*, i32, i8*, i32, i8*, i32, i8*)*, i32 (%struct._IO_FILE*, i8*, i8*, %struct._PyArray_Descr*)*, i32 (i8*, i8*, i8**, %struct._PyArray_Descr*)*, i8 (i8*, i8*)*, i32 (i8*, i32, i8*)*, i32 (i8*, i32, i8*, i8*)*, [3 x i32 (i8*, i32, i8*)*], [3 x i32 (i8*, i32*, i32, i8*)*], %struct._object*, i32 (i8*)*, i32**, i32*, void (i8*, i32, i8*, i8*, i8*)*, void (i8*, i8*, i32, i8*, i32)*, i32 (i8*, i8*, i32*, i32, i32, i32, i32, i32)* }

; Unboxing external declarations
declare i32 @PyInt_AsLong(%struct._object*)
"""

    _wrapper = """
define void @wrap_%(kernel_name)s__(%%struct._object* %%_start,
                                    %%struct._object* %%_end,
                                    %(wrapper_args)s) nounwind {
    %%start = call i32 @PyInt_AsLong(%%struct._object* %%_start)
    %%end = call i32 @PyInt_AsLong(%%struct._object* %%_end)
    %%i = alloca i32, align 4
    %%n = alloca i32, align 4
    %(wrapper_decs)s

    store i32 %%start, %%n, align 4
    br label %%loop_cond

loop_cond:
    %%curr_n = load i32* %%n, align 4
    %%n_lt_end = icmp slt i32 %%curr_n, %%end
    br i1 %%n_lt_end, label %%loop_body, label %%loop_end

loop_body:
    %(index_expr)s                          ; Must place result into %%curr_i
    store i32 %%curr_i, %%i, align 4
    %(itset_loop_body)s

    %%next_n = add nsw %%curr_n, 1
    store i32 %%next_n, i32* %%n, align 4
    br label %%loop_cond

loop_end:
    ret void
}
"""

    def compile(self):
        pass

    def generate_code(self):
        pass


class Arg(host.Arg):
    pass


class ParLoop(host.ParLoop):
    pass


def _setup():
    pass
