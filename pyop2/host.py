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

from __future__ import absolute_import, print_function, division

from copy import deepcopy

import numpy

import coffee.system
from coffee.plan import ASTKernel

from pyop2 import base, compilation
from pyop2.base import *
# Override base ParLoop with flop-logging version in petsc_base
from pyop2.petsc_base import ParLoop  # noqa: pass-through
from pyop2.mpi import collective
from pyop2.configuration import configuration
from pyop2.utils import as_tuple
from pyop2.wrapper import (ArgWrapper, DirectLayerAccess,
                           IncrementalLayerLoop, List, Singleton,
                           Slice, add, concat, deref, _map_vec, _indices)


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
        block_row = (%(rowmap)s)[j];
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
        block_col = (%(colmap)s)[j];
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

    rmap_ = Slice(local_maps[0].value_type, "rowmap", rmap.arity * dim[0])
    cmap_ = Slice(local_maps[1].value_type, "colmap", cmap.arity * dim[1])
    return bcs_ops, "MatSetValuesLocal", (rmap_, cmap_)


class Kernel(base.Kernel):

    def _ast_to_c(self, ast, opts={}):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of code (C syntax) suitable to CPU execution."""
        self._original_ast = deepcopy(ast)
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

    def init_and_writeback(self, args, c, col, namer, is_facet=False, start_layer=None):
        if isinstance(self.data, Mat):
            assert self.idx is not None
            assert not self._is_mixed_mat

            mat_name = args[0]
            map_names = args[1:]
            buf_name = namer('buf')

            dim = self.data.dims[0][0]  # TODO

            if start_layer is None:
                layer = DirectLayerAccess(col)
            else:
                layer = IncrementalLayerLoop(start_layer, namer)

            map_vecs, offsets = zip(*[_map_vec(name, m.arity, m.offset, idx, c, is_facet=is_facet)
                                      for m, name, idx in zip(self.map, map_names, self.idx)])
            arity = [m.size for m in map_vecs]

            assert len(arity) == len(dim)
            size = [n * d for n, d in zip(arity, dim)]

            init = [str.format("double {buf}[{s1}][{s2}] __attribute__((aligned(16))) = {{{{0.0}}}};",
                               buf=buf_name, s1=size[0], s2=size[1])]  # TODO
            writeback = []

            w_init = []
            w_init_layer = []
            w_writeback = []
            w_post_writeback = []
            local_maps = []
            for r in range(2):
                def use(direct):
                    name_thunk = lambda: namer('local_map' + str(r))
                    writeback, local_map = direct.as_slice(name_thunk)
                    post_writeback = []

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
                        post_writeback.append("if ({col} == 0) {{".format(col=col))
                        for i, neg in enumerate(bottom_mask):
                            if neg < 0:
                                writeback.append("\t{lmap_name}[{i}] -= 10000000;".format(lmap_name=local_map.expr, i=i))
                                post_writeback.append("\t{lmap_name}[{i}] += 10000000;".format(lmap_name=local_map.expr, i=i))
                        writeback.append("}")
                        post_writeback.append("}")
                    if any(top_mask):
                        top_layer = "(nlayers - 1)" if is_facet else "nlayers"
                        writeback.append("if ({col} == {top_layer} - 1) {{".format(col=col, top_layer=top_layer))
                        post_writeback.append("if ({col} == {top_layer} - 1) {{".format(col=col, top_layer=top_layer))
                        for i, neg in enumerate(top_mask):
                            if neg < 0:
                                writeback.append("\t{lmap_name}[{i}] -= 10000000;".format(lmap_name=local_map.expr,
                                                                                          i=(m.arity + i if is_facet else i)))
                                post_writeback.append("\t{lmap_name}[{i}] += 10000000;".format(lmap_name=local_map.expr,
                                                                                               i=(m.arity + i if is_facet else i)))
                        writeback.append("}")
                        post_writeback.append("}")
                    return ArgWrapper(writeback=writeback, post_writeback=post_writeback), local_map
                arg_wrapper, local_map = layer(use, map_vecs[r], offsets[r])
                w_init.extend(arg_wrapper.init)
                w_init_layer.extend(arg_wrapper.init_layer)
                w_writeback.extend(arg_wrapper.writeback)
                w_post_writeback.extend(arg_wrapper.post_writeback)
                local_maps.append(local_map)

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

            if all(offset is None for offset in offsets):
                return ArgWrapper(init=init+w_init, init_layer=w_init_layer, writeback=w_writeback+writeback, post_writeback=w_post_writeback), buf_name
            else:
                return ArgWrapper(init=w_init, init_layer=init+w_init_layer, writeback=w_writeback+writeback, post_writeback=w_post_writeback), buf_name

        elif isinstance(self.data, Dat) and self.map is not None:
            assert len(self.data) == len(self.map)
            M = len(self.data)
            dat_names = args[:M]
            map_names = args[M:]

            buf_name = namer('vec')

            if start_layer is None:
                layer = DirectLayerAccess(col)
            else:
                layer = IncrementalLayerLoop(start_layer, namer)

            pointers = []
            offsets = []
            for dat_name, map_name, dat, map_ in zip(dat_names, map_names, self.data, self.map):
                map_vec, offset = _map_vec(map_name, map_.arity,
                                           map_.offset, self.idx, c,
                                           is_facet=is_facet)
                if offset is None:
                    offset = [0] * map_.arity
                offset = numpy.array(offset) * dat.cdim
                if self.idx is not None or self._flatten:
                    if self._flatten:
                        offsets.extend(numpy.hstack([offset] * dat.cdim).flat)
                    else:
                        offsets.extend(numpy.vstack([offset] * dat.cdim).transpose().flat)
                else:
                    offsets.extend(offset)
                indices = _indices(dat.cdim, map_vec, flatten=self._flatten)
                if self.idx is None and not self._flatten:
                    # Special case: reduced buffer length
                    indices = List(indices.value_type, indices.as_list().values[::dat.cdim])
                g_dat = Singleton("{0}*".format(self.data.ctype), dat_name)
                pointers.append(add(g_dat, indices))
            pointers = concat(*pointers)

            if self.idx is None:
                def use(direct):
                    init, kernel_buf = direct.as_slice(lambda: buf_name)
                    return ArgWrapper(init=init), kernel_buf.expr
                return layer(use, pointers, offsets)

            if isinstance(self.idx, IterationIndex):
                assert self.idx.index == 0

            def use(direct):
                lvalues = deref(direct)

                if isinstance(lvalues, Slice):
                    return ArgWrapper(), lvalues.expr

                if self.access in [READ, RW]:
                    init, buf_slice = lvalues.as_slice(lambda: buf_name)
                elif self.access in [WRITE, INC]:
                    # TSFC expects zero buffer for WRITE, too.
                    init = [str.format("{typename} {buf}[{size}] = {{0.0}};",
                                       typename=self.data.ctype, buf=buf_name,
                                       size=lvalues.size)]
                else:
                    raise NotImplementedError("Access descriptor {0} not implemented".format(self.access))

                writeback = []
                if self.access in [RW, WRITE, INC]:
                    op = '='
                    if self.access == INC:
                        op = '+='

                    for i, lvalue in enumerate(lvalues.as_list().values):
                        writeback.append(str.format("{lvalue} {op} {buf_name}[{i}];",
                                                    buf_name=buf_name, i=i,
                                                    lvalue=lvalue, op=op))

                return ArgWrapper(init=init, writeback=writeback), buf_name
            return layer(use, pointers, offsets)
        elif isinstance(self.data, DatView) and self.map is None:
            dat_name, = args
            kernel_arg = "{dat} + {c} * {dim} + {i}".format(dat=dat_name, c=c, dim=super(DatView, self.data).cdim, i=self.data.index)
            return ArgWrapper(), kernel_arg
        elif isinstance(self.data, Dat) and self.map is None:
            dat_name, = args
            kernel_arg = "{dat} + {c} * {dim}".format(dat=dat_name, c=c, dim=self.data.cdim)
            return ArgWrapper(), kernel_arg
        elif isinstance(self.data, Global):
            arg_name, = args
            return ArgWrapper(), arg_name
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
        self._cppargs = deepcopy(type(self)._cppargs)
        self._libraries = deepcopy(type(self)._libraries)
        self._system_headers = deepcopy(type(self)._system_headers)
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
