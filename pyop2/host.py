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

from collections import namedtuple
from copy import deepcopy as dcopy
import numpy

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


integer_types = {"int"}


class Singleton(namedtuple('Singleton', ['value_type', 'value'])):
    def repeat(self, n):
        return List(self.value_type, [self.value] * n)


class List(namedtuple('List', ['value_type', 'values'])):
    @property
    def size(self):
        return len(self.values)

    def as_list(self):
        return self

    def as_slice(self, name_thunk):
        buf = name_thunk()
        statements = ["{0} {1}[{2}];".format(self.value_type, buf, self.size)]
        for i, expr in enumerate(self.values):
            statements.append("{0}[{1}] = {2};".format(buf, i, expr))
        return statements, Slice(self.value_type, buf, self.size)


class Range(namedtuple('Range', ['value_type', 'expr', 'size'])):
    def as_list(self):
        return List(self.value_type,
                    ['{0} + {1}'.format(self.expr, n)
                     for n in range(self.size)])

    def as_slice(self, name_thunk):
        return self.as_list().as_slice(name_thunk)


class Slice(namedtuple('Slice', ['value_type', 'expr', 'size'])):
    def as_list(self):
        return List(self.value_type,
                    ['({0})[{1}]'.format(self.expr, n)
                     for n in range(self.size)])

    def as_slice(self, name_thunk):
        return [], self


def deref(expr_vec):
    if not expr_vec.value_type.endswith('*'):
        raise ValueError("Can only dereference pointer types, not {0}".format(expr_vec.value_type))
    value_type = expr_vec.value_type[:-1]  # drop star

    if isinstance(expr_vec, Range):
        return Slice(value_type, expr_vec.expr, expr_vec.size)
    else:
        expr_vec = expr_vec.as_list()
        return List(value_type, ["*({0})".format(e) for e in expr_vec.values])


def add(x, y):
    if x.value_type == y.value_type and x.value_type in integer_types:
        value_type = x.value_type
    elif x.value_type.endswith('*') and y.value_type in integer_types:
        value_type = x.value_type
    elif y.value_type.endswith('*') and x.value_type in integer_types:
        value_type = y.value_type
    else:
        raise RuntimeError("Type mismatch: '{0}' + '{1}'".format(x.value_type, y.value_type))

    if isinstance(x, Singleton) and isinstance(y, Singleton):
        return Singleton(value_type, "{0} + {1}".format(x.value, y.value))

    if isinstance(x, Range) and isinstance(y, Singleton):
        x, y = y, x

    if isinstance(x, Singleton) and isinstance(y, Range):
        return Range(value_type, "{0} + {1}".format(x.value, y.expr), y.size)

    if isinstance(x, Singleton):
        x = x.repeat(y.size)
    else:
        x = x.as_list()

    if isinstance(y, Singleton):
        y = y.repeat(x.size)
    else:
        y = y.as_list()

    return List(value_type, ["{0} + {1}".format(a, b)
                             for a, b in zip(x.values, y.values)])


def _map_vec(map_name, arity, offset, iteration_index, element_index, column_index, is_facet=False):
    g_map = Singleton("int*", map_name)
    l_map = deref(add(g_map, Range("int", "{0}*{1}".format(element_index, arity), arity)))
    if isinstance(iteration_index, int):
        l_map = List(l_map.value_type, [l_map.as_list().values[iteration_index]])

    if offset is not None and any(offset):
        assert column_index is not None
        assert arity == len(offset)
        lb_map = l_map

        if isinstance(iteration_index, int):
            offset = [offset[iteration_index]]

        offset_list = List("int", ["{0}*{1}".format(column_index, offset[r])
                                   for r in range(len(offset))])
        l_map = add(lb_map, offset_list)

        if is_facet:
            offset1_list = List("int",
                                ["({0} + 1)*{1}".format(column_index, offset[r])
                                 for r in range(len(offset))])
            l1_map = add(lb_map, offset1_list)
            l_map = List(l_map.value_type,
                         l_map.as_list().values + l1_map.as_list().values)

    return l_map


def _pointers(dat_name, dim, map_vec, flatten=False):
    if map_vec.size == 1:
        start = "({0})*{1}".format(map_vec.as_list().values[0], dim)
        indices = Range(map_vec.value_type, start, dim)
    elif dim == 1:
        indices = map_vec
    else:
        if flatten:
            ordering = ((i, d) for d in range(dim) for i in range(map_vec.size))
        else:
            ordering = ((i, d) for i in range(map_vec.size) for d in range(dim))
        map_vec = map_vec.as_list()
        indices = List(map_vec.value_type,
                       [str.format("({map_item})*{dim} + {d}",
                                   map_item=map_vec.values[i],
                                   dim=dim, d=d)
                        for i, d in ordering])

    g_dat = Singleton("double*", dat_name)  # FIXME: C typename
    return add(g_dat, indices).as_list().values


def vfs_component_bcs(maps, dim, local_maps):
    rmap, cmap = maps
    if rmap.vector_index is None and cmap.vector_index is None:
        # Nothing to do here
        return [], "MatSetValuesBlockedLocal", local_maps

    # Horrible hack alert
    # To apply BCs to a component of a Dat with cdim > 1
    # we encode which components to apply things to in the
    # high bits of the map value
    # The value that comes in is:
    # -(row + 1 + sum_i 2 ** (30 - i))
    # where i are the components to zero
    #
    # So, the actual row (if it's negative) is:
    # (~input) & ~0x70000000
    # And we can determine which components to zero by
    # inspecting the high bits (1 << 30 - i)
    template = """
    PetscInt rowmap[%(nrows)d*%(rdim)d];
    PetscInt colmap[%(ncols)d*%(cdim)d];
    int discard, tmp, block_row, block_col;
    for ( int j = 0; j < %(nrows)d; j++ ) {
        block_row = %(rowmap)s[i*%(nrows)d + j];
        discard = 0;
        if ( block_row < 0 ) {
            tmp = -(block_row + 1);
            discard = 1;
            block_row = tmp & ~0x70000000;
        }
        for ( int k = 0; k < %(rdim)d; k++ ) {
            if ( discard && (%(drop_full_row)d || ((tmp & (1 << (30 - k))) != 0)) ) {
                rowmap[j*%(rdim)d + k] = -1;
            } else {
                rowmap[j*%(rdim)d + k] = (block_row)*%(rdim)d + k;
            }
        }
    }
    for ( int j = 0; j < %(ncols)d; j++ ) {
        discard = 0;
        block_col = %(colmap)s[i*%(ncols)d + j];
        if ( block_col < 0 ) {
            tmp = -(block_col + 1);
            discard = 1;
            block_col = tmp & ~0x70000000;
        }
        for ( int k = 0; k < %(cdim)d; k++ ) {
            if ( discard && (%(drop_full_col)d || ((tmp & (1 << (30 - k))) != 0)) ) {
                colmap[j*%(rdim)d + k] = -1;
            } else {
                colmap[j*%(cdim)d + k] = (block_col)*%(cdim)d + k;
            }
        }
    }
    """
    fdict = {'nrows': rmap.arity,
             'ncols': cmap.arity,
             'rdim': dim[0],
             'cdim': dim[1],
             'rowmap': local_maps[0].expr,
             'colmap': local_maps[1].expr,
             'drop_full_row': 0 if rmap.vector_index is not None else 1,
             'drop_full_col': 0 if cmap.vector_index is not None else 1}
    bcs_ops = [template % fdict]

    rmap_ = Slice(rmap.value_type, "rowmap", rmap.arity * dim[0])
    cmap_ = Slice(cmap.value_type, "colmap", cmap.arity * dim[1])
    return bcs_ops, "MatSetValuesLocal", (rmap_, cmap_)


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

    def init_and_writeback(self, args, c, col, namer, is_facet=False):
        if isinstance(self.data, Mat):
            assert self.idx is not None
            assert not self._is_mixed_mat

            mat_name = args[0]
            map_names = args[1:]
            buf_name = namer('buf')

            dim = self.data.dims[0][0]  # TODO

            map_vecs = [_map_vec(name, m.arity, m.offset, idx, c, col, is_facet=is_facet)
                        for m, name, idx in zip(self.map, map_names, self.idx)]
            arity = [m.size for m in map_vecs]

            assert len(arity) == len(dim)
            size = [n * d for n, d in zip(arity, dim)]

            init = [str.format("double {buf}[{s1}][{s2}] __attribute__((aligned(16))) = {{{{0.0}}}};",
                               buf=buf_name, s1=size[0], s2=size[1])]  # TODO
            writeback = []

            local_maps = []
            for r in range(2):
                name_thunk = lambda: namer('local_map' + str(r))
                slice_init, local_map = map_vecs[r].as_slice(name_thunk)
                writeback.extend(slice_init)
                local_maps.append(local_map)

                m = self.map[r]
                bottom_mask = numpy.zeros(m.arity)
                top_mask = numpy.zeros(m.arity)
                for location, name in m.implicit_bcs:
                    if location == "bottom":
                        bottom_mask += m.bottom_mask[name]
                    elif location == "top":
                        top_mask += m.top_mask[name]
                if any(bottom_mask):
                    writeback.append("if ({col} == 0) {{".format(col=col))
                    for i, neg in enumerate(bottom_mask):
                        if neg < 0:
                            writeback.append("\t{lmap_name}[{i}] = -1;".format(lmap_name=local_map.expr, i=i))
                    writeback.append("}")
                if any(top_mask):
                    writeback.append("if ({col} == top_layer - 1) {{".format(col=col))
                    for i, neg in enumerate(top_mask):
                        if neg < 0:
                            writeback.append("\t{lmap_name}[{i}] = -1;".format(lmap_name=local_map.expr,
                                                                               i=(m.arity + i if is_facet else i)))
                    writeback.append("}")

            if self._flatten and any(a > 1 and d > 1 for a, d in zip(arity, dim)):
                ins_name = namer('ins')
                writeback.append(str.format("double {ins}[{s1}][{s2}] __attribute__((aligned(16)));",
                                            ins=ins_name, s1=size[0], s2=size[1]))  # TODO

                for j in range(arity[0]):
                    for k in range(dim[0]):
                        for l in range(arity[1]):
                            for m in range(dim[1]):
                                line = str.format("{0}[{2}][{3}] = {1}[{4}][{5}];",
                                                  ins_name, buf_name,
                                                  dim[0]*j + k, dim[1]*l + m,
                                                  arity[0]*k + j, arity[1]*m + l)
                                writeback.append(line)
            else:
                ins_name = buf_name

            # VFS component BCs
            bcs_ops, mat_func, local_maps = vfs_component_bcs(self.map, dim, local_maps)
            writeback.extend(bcs_ops)

            # Writeback
            template = "{mat_func}({mat}, {map1_size}, {map1_expr}, {map2_size}, {map2_expr}, (const PetscScalar *){ins}, {mode});"
            writeback.append(template.format(
                mat_func=mat_func,
                mat=mat_name, ins=ins_name,
                map1_size=local_maps[0].size,
                map1_expr=local_maps[0].expr,
                map2_size=local_maps[1].size,
                map2_expr=local_maps[1].expr,
                mode={WRITE: "INSERT_VALUES", INC: "ADD_VALUES"}[self.access]))

            return init, writeback, buf_name

        elif isinstance(self.data, Dat) and self.map is not None:
            assert len(self.data) == len(self.map)
            M = len(self.data)
            dat_names = args[:M]
            map_names = args[M:]

            pointers = []

            init = []
            writeback = []

            buf_name = namer('vec')

            for dat_name, map_name, dat, map_ in zip(dat_names, map_names, self.data, self.map):
                map_vec = _map_vec(map_name, map_.arity, map_.offset, self.idx, c, col, is_facet=is_facet)
                pointers_ = _pointers(dat_name, dat.cdim, map_vec, flatten=self._flatten)
                if self.idx is None and not self._flatten:
                    # Special case: reduced buffer length
                    pointers_ = pointers_[::dat.cdim]
                pointers.extend(pointers_)

            if self.idx is None:
                init.append("{typename} *{buf}[{size}];".format(typename=self.data.ctype, buf=buf_name, size=len(pointers)))
                for i, pointer in enumerate(pointers):
                    init.append("{buf_name}[{i}] = {pointer};".format(buf_name=buf_name, i=i, pointer=pointer))

            else:
                if isinstance(self.idx, IterationIndex):
                    assert self.idx.index == 0

                initializer = ''
                if self.access in [WRITE, INC]:  # TSFC expects zero buffer for WRITE
                    initializer = ' = {0.0}'
                init.append("{typename} {buf}[{size}]{initializer};".format(typename=self.data.ctype, buf=buf_name, size=len(pointers), initializer=initializer))

                if self.access in [READ, RW]:
                    for i, pointer in enumerate(pointers):
                        init.append("{buf_name}[{i}] = *({pointer});".format(buf_name=buf_name, i=i, pointer=pointer))

                if self.access in [RW, WRITE, INC]:
                    op = '='
                    if self.access == INC:
                        op = '+='

                    for i, pointer in enumerate(pointers):
                        writeback.append("*({pointer}) {op} {buf_name}[{i}];".format(buf_name=buf_name, i=i, pointer=pointer, op=op))

                if self.access not in [READ, WRITE, RW, INC]:
                    raise NotImplementedError("Access descriptor {0} not implemented".format(self.access))

            return init, writeback, buf_name
        elif isinstance(self.data, DatView) and self.map is None:
            dat_name, = args
            kernel_arg = "{dat} + {c} * {dim} + {i}".format(dat=dat_name, c=c, dim=super(DatView, self.data).cdim, i=self.data.index)
            return [], [], kernel_arg
        elif isinstance(self.data, Dat) and self.map is None:
            dat_name, = args
            kernel_arg = "{dat} + {c} * {dim}".format(dat=dat_name, c=c, dim=self.data.cdim)
            return [], [], kernel_arg
        elif isinstance(self.data, Global):
            arg_name, = args
            return [], [], arg_name
        else:
            raise NotImplementedError("How to handle {0}?".format(type(self.data).__name__))


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

        code_to_compile = """
        #include <petsc.h>
        #include <stdbool.h>
        #include <math.h>
        %(sys_headers)s

        %(kernel)s

        %(externc_open)s
        %(wrapper)s
        %(externc_close)s
        """ % {'kernel': kernel_code,
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
