# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Base classes extending those from the :mod:`base` module with functionality
common to backends executing on the host."""

from copy import deepcopy as dcopy
from itertools import product

import base
import compilation
from base import *
# Override base ParLoop with flop-logging version in petsc_base
from petsc_base import ParLoop  # noqa: pass-through
from mpi import collective
from configuration import configuration
from utils import as_tuple

import coffee.system
from coffee.plan import ASTKernel


class Kernel(base.Kernel):

    def _ast_to_c(self, ast, opts={}):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of code (C syntax) suitable to CPU execution."""
        self._original_ast = dcopy(ast)
        ast_handler = ASTKernel(ast, self._include_dirs)
        ast_handler.plan_cpu(self._opts)
        return ast_handler.gencode()


class Arg(base.Arg):

    def wrapper_args(self):
        # TODO: Use cache key to calculate types.
        c_typenames = []
        types = []
        values = []
        if self._is_mat:
            c_typenames.append("Mat")
            types.append(self.data._argtype)
            values.append(self.data.handle.handle)
        else:
            for d in self.data:
                c_typenames.append(self.ctype)
                types.append(d._argtype)
                # Cannot access a property of the Dat or we will force
                # evaluation of the trace
                values.append(d._data.ctypes.data)
        if self._is_indirect or self._is_mat:
            maps = as_tuple(self.map, Map)
            for map in maps:
                for m in map:
                    c_typenames.append("int")
                    types.append(m._argtype)
                    values.append(m._values.ctypes.data)
        return c_typenames, types, values

    def init_and_writeback(self, args, c, namer):
        if isinstance(self.data, Mat):
            assert self._flatten
            assert self.idx is not None

            arity = [m.arity for m in self.map]
            dim = self.data.dims[0][0]

            assert len(arity) == len(dim)
            size = [n * d for n, d in zip(arity, dim)]

            mat_name, map1_name, map2_name = args
            buf_name = namer('buffer')

            init = ["double {buf}[{size1}][{size2}] __attribute__((aligned(16))) = {{0.0}};".format(
                buf=buf_name, size1=size[0], size2=size[1])]

            if all(d == 1 for d in dim):
                writeback = ["""MatSetValuesLocal({mat}, {arity1}, {map1} + {c} * {arity1},
\t\t\t{arity2}, {map2} + {c} * {arity2},
\t\t\t(const PetscScalar *){buf},
\t\t\tADD_VALUES);""".format(mat=mat_name, buf=buf_name, arity1=arity[0], arity2=arity[1], map1=map1_name, map2=map2_name, c=c)]
            else:
                tmp_name = namer('tmp_buffer')

                idx = "[%(ridx)s][%(cidx)s]"
                idx_l = idx % {'ridx': "%d*j + k" % dim[0],
                               'cidx': "%d*l + m" % dim[1]}
                idx_r = idx % {'ridx': "j + %d*k" % arity[0],
                               'cidx': "l + %d*m" % arity[1]}
                # Shuffle xxx yyy zzz into xyz xyz xyz
                writeback = ["""
                double %(tmp_name)s[%(size1)d][%(size2)d] __attribute__((aligned(16)));
                for ( int j = 0; j < %(nrows)d; j++ ) {
                   for ( int k = 0; k < %(rbs)d; k++ ) {
                      for ( int l = 0; l < %(ncols)d; l++ ) {
                         for ( int m = 0; m < %(cbs)d; m++ ) {
                            %(tmp_name)s%(idx_l)s = %(buf_name)s%(idx_r)s;
                         }
                      }
                   }
                }""" % {'nrows': arity[0],
                        'ncols': arity[1],
                        'rbs': dim[0],
                        'cbs': dim[1],
                        'idx_l': idx_l,
                        'idx_r': idx_r,
                        'buf_name': buf_name,
                        'tmp_name': tmp_name,
                        'size1': size[0],
                        'size2': size[1]}]

                writeback += ["""MatSetValuesBlockedLocal({mat}, {arity1}, {map1} + {c} * {arity1},
\t\t\t{arity2}, {map2} + {c} * {arity2},
\t\t\t(const PetscScalar *){tmp_name},
\t\t\tADD_VALUES);""".format(mat=mat_name, tmp_name=tmp_name, arity1=arity[0], arity2=arity[1], map1=map1_name, map2=map2_name, c=c)]

            return init, writeback, buf_name

        elif isinstance(self.data, Dat) and self.map is not None:
            dat_name, map_name = args
            buf_name = namer('vec')

            init = []
            writeback = []

            if self.idx is not None:
                assert isinstance(self.idx, IterationIndex) and self.idx.index == 0
                star = ''
                value_of = lambda x: "*({0})".format(x)
            else:
                star = '*'
                value_of = lambda x: x

            pointers = _pointers(dat_name, map_name, self.map.arity, self.data.cdim, c, flatten=self._flatten)
            if self.idx is None and not self._flatten:
                # Special case: reduced buffer length
                pointers = pointers[::self.data.cdim]

            init.append("{typename} {star}{buf}[{size}]{init};".format(typename=self.data.ctype, star=star, buf=buf_name, size=len(pointers), init=(' = {0.0}' if self.access == INC else '')))

            if self.access == READ:
                for i, pointer in enumerate(pointers):
                    init.append("{buf_name}[{i}] = {value};".format(buf_name=buf_name, i=i, value=value_of(pointer)))

            elif self.access in [WRITE, INC]:
                if self.idx:
                    op = {WRITE: '=', INC: '+='}[self.access]
                    for i, pointer in enumerate(pointers):
                        writeback.append("{value} {op} {buf_name}[{i}];".format(buf_name=buf_name, i=i, value=value_of(pointer), op=op))
                else:
                    for i, pointer in enumerate(pointers):
                        init.append("{buf_name}[{i}] = {pointer};".format(buf_name=buf_name, i=i, pointer=pointer))

            elif self.access == RW:
                for i, pointer in enumerate(pointers):
                    init.append("{buf_name}[{i}] = {value};".format(buf_name=buf_name, i=i, value=value_of(pointer)))

                if self.idx:
                    for i, pointer in enumerate(pointers):
                        writeback.append("{value} = {buf_name}[{i}];".format(buf_name=buf_name, i=i, value=value_of(pointer)))

            else:
                raise NotImplementedError("Access descriptor {0} not implemented".format(self.access))

            return init, writeback, buf_name
        elif isinstance(self.data, Dat) and self.map is None:
            dat_name, = args
            kernel_arg = "{dat} + {c} * {dim}".format(dat=dat_name, c=c, dim=self.data.cdim)
            return [], [], kernel_arg
        elif isinstance(self.data, Global):
            arg_name, = args
            return [], [], arg_name
        else:
            raise NotImplementedError("How to handle {0}?".format(type(self.data).__name__))


def _pointers(dat_name, map_name, arity, dim, i, flatten):
    template = "{dat_name} + {map_name}[{i} * {arity} + {r}] * {dim} + {d}"
    if flatten:
        ordering = ((r, d) for d, r in product(range(dim), range(arity)))
    else:
        ordering = ((r, d) for r, d in product(range(arity), range(dim)))
    return [template.format(dat_name=dat_name, map_name=map_name,
                            arity=arity, dim=dim, i=i, r=r, d=d)
            for r, d in ordering]


class JITModule(base.JITModule):

    _cppargs = []
    _libraries = []
    _system_headers = []
    _extension = 'c'

    def __init__(self, kernel, itspace, *args, **kwargs):
        """
        A cached compiled function to execute for a specified par_loop.

        See :func:`~.par_loop` for the description of arguments.

        .. warning ::

           Note to implementors.  This object is *cached*, and therefore
           should not hold any long term references to objects that
           you want to be collected.  In particular, after the
           ``args`` have been inspected to produce the compiled code,
           they **must not** remain part of the object's slots,
           otherwise they (and the :class:`~.Dat`\s, :class:`~.Map`\s
           and :class:`~.Mat`\s they reference) will never be collected.
        """
        # Return early if we were in the cache.
        if self._initialized:
            return
        self.comm = itspace.comm
        self._kernel = kernel
        self._fun = None
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)
        self._iteration_region = kwargs.get('iterate', ALL)
        self._initialized = True
        # Copy the class variables, so we don't overwrite them
        self._cppargs = dcopy(type(self)._cppargs)
        self._libraries = dcopy(type(self)._libraries)
        self._system_headers = dcopy(type(self)._system_headers)
        self.set_argtypes(itspace.iterset, *args)
        self.compile()

    @collective
    def __call__(self, *args):
        return self._fun(*args)

    @property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        compiler = coffee.system.compiler
        externc_open = '' if not self._kernel._cpp else 'extern "C" {'
        externc_close = '' if not self._kernel._cpp else '}'
        headers = "\n".join([compiler.get('vect_header', "")])
        if any(arg._is_soa for arg in self._args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            %(header)s
            %(code)s
            #undef OP2_STRIDE
            """ % {'code': self._kernel.code(),
                   'header': headers}
        else:
            kernel_code = """
            %(header)s
            %(code)s
            """ % {'code': self._kernel.code(),
                   'header': headers}
        code_to_compile = self.generate_wrapper()

        _const_decs = '\n'.join([const._format_declaration()
                                for const in Const._definitions()]) + '\n'

        code_to_compile = """
        #include <petsc.h>
        #include <stdbool.h>
        #include <math.h>
        %(sys_headers)s
        %(consts)s

        %(kernel)s

        %(externc_open)s
        %(wrapper)s
        %(externc_close)s
        """ % {'consts': _const_decs, 'kernel': kernel_code,
               'wrapper': code_to_compile,
               'externc_open': externc_open,
               'externc_close': externc_close,
               'sys_headers': '\n'.join(self._kernel._headers + self._system_headers)}

        self._dump_generated_code(code_to_compile)
        if configuration["debug"]:
            self._wrapper_code = code_to_compile

        extension = self._extension
        cppargs = self._cppargs
        cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
                   ["-I%s" % d for d in self._kernel._include_dirs] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        if compiler:
            cppargs += [compiler[coffee.system.isa['inst_set']]]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        ldargs += self._kernel._ldargs

        if self._kernel._cpp:
            extension = "cpp"
        self._fun = compilation.load(code_to_compile,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=cppargs,
                                     ldargs=ldargs,
                                     argtypes=self._argtypes,
                                     restype=None,
                                     compiler=compiler.get('name'),
                                     comm=self.comm)
        # Blow away everything we don't need any more
        del self._args
        del self._kernel
        del self._itspace
        del self._direct
        return self._fun

    def generate_wrapper(self):
        raise NotImplementedError("How to generate a wrapper?")
