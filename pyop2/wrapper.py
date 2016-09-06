# -*- coding: utf-8 -*-
#
# Copyright (c) 2016, Miklós HOMOLYA
# All rights reserved.
#
# This file is part of PyOP2
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Utility functions for generating wrappers for PyOP2 kernels."""

from __future__ import absolute_import, print_function, division

from collections import namedtuple

import numpy

from pyop2.base import READ, WRITE, RW, INC
from pyop2.base import Dat, DatView, Global, IterationIndex, Mat


integer_types = {"int"}
"""Set of C typenames recognised as integer types"""


class Singleton(namedtuple('Singleton', ['value_type', 'value'])):
    """A single value with type"""

    def repeat(self, n):
        """Construct a :py:class:`List` by repeating this value ``n`` times."""
        return List(self.value_type, [self.value] * n)


class List(namedtuple('List', ['value_type', 'values'])):
    """An array of (unrelated) values of the same type"""

    @property
    def size(self):
        """Array length"""
        return len(self.values)

    def as_list(self):
        """Cast to :py:class:`List`."""
        return self  # no-op

    def as_slice(self, name_thunk):
        """Cast to :py:class:`Slice`.

        :arg name_thunk: A thunk that returns a buffer name.
        :returns: List of operations to fill the buffer, and a
                  :py:class:`Slice` object.
        """
        buf = name_thunk()
        statements = ["{0} {1}[{2}];".format(self.value_type, buf, self.size)]
        for i, expr in enumerate(self.values):
            statements.append("{0}[{1}] = {2};".format(buf, i, expr))
        return statements, Slice(self.value_type, buf, self.size)


class Range(namedtuple('Range', ['value_type', 'expr', 'size'])):
    """An array of consecutive integral values of the same type.

    For example, [12, 13, 14, 15] represented as (expr=12, size=4).
    """
    def as_list(self):
        """Cast to :py:class:`List`."""
        return List(self.value_type,
                    ['{0} + {1}'.format(self.expr, n)
                     for n in range(self.size)])

    def as_slice(self, name_thunk):
        """Cast to :py:class:`Slice`.

        :arg name_thunk: A thunk that returns a buffer name.
        :returns: List of operations to fill the buffer, and a
                  :py:class:`Slice` object.
        """
        return self.as_list().as_slice(name_thunk)


class Slice(namedtuple('Slice', ['value_type', 'expr', 'size'])):
    """An array in memory, i.e. an array of lvalues with consecutive
    locations in memory."""

    def as_list(self):
        """Cast to :py:class:`List`."""
        return List(self.value_type,
                    ['({0})[{1}]'.format(self.expr, n)
                     for n in range(self.size)])

    def as_slice(self, name_thunk):
        """Cast to :py:class:`Slice`.

        :arg name_thunk: A thunk that returns a buffer name (not called).
        :returns: Empty list and ``self``.
        """
        return [], self  # no-op


class ArgWrapper(object):
    def __init__(self, init=None, init_layer=None, writeback=None, post_writeback=None):
        if init is None:
            init = []
        if init_layer is None:
            init_layer = []
        if writeback is None:
            writeback = []
        if post_writeback is None:
            post_writeback = []

        self.init = init
        self.init_layer = init_layer
        self.writeback = writeback
        self.post_writeback = post_writeback


class DirectLayerAccess(object):
    def __init__(self, layer_index):
        self.layer_index = layer_index

    def __call__(self, use, base, offset):
        if offset is not None and any(offset):
            assert len(offset) == base.size
            column = ["{j}*{off}".format(j=self.layer_index, off=off)
                      for i, off in enumerate(offset)]
            arg_wrapper, payload = use(add(base, List("int", column)))
            assert not arg_wrapper.init_layer

            # Move init to init_layer
            arg_wrapper.init_layer = arg_wrapper.init
            arg_wrapper.init = []
            return arg_wrapper, payload
        else:
            arg_wrapper, payload = use(base)
            return arg_wrapper, payload


class IncrementalLayerLoop(object):
    def __init__(self, start_layer, unique_name):
        self.start_layer = start_layer
        self.unique_name = unique_name

    def __call__(self, use, base, offset):
        if offset is not None and any(offset):
            assert len(offset) == base.size
            name_thunk = lambda: self.unique_name('direct')
            if self.start_layer == 0:
                init, direct = base.as_list().as_slice(name_thunk)
            else:
                start = ["{j}*{off}".format(j=self.start_layer, off=off)
                         for i, off in enumerate(offset)]
                init, direct = add(base, List("int", start)).as_slice(name_thunk)
            arg_wrapper, payload = use(direct)
            assert not arg_wrapper.init_layer

            post_writeback = arg_wrapper.post_writeback[:]
            post_writeback += ["{0}[{1}] += {2};".format(direct.expr, i, off)
                               for i, off in enumerate(offset)]
            return ArgWrapper(init=init,
                              init_layer=arg_wrapper.init,
                              writeback=arg_wrapper.writeback,
                              post_writeback=post_writeback), payload
        else:
            arg_wrapper, payload = use(base)
            return arg_wrapper, payload


def deref(expr_vec):
    """Dereference each element of an array expression."""
    if not expr_vec.value_type.endswith('*'):
        raise ValueError("Can only dereference pointer types, not {0}".format(expr_vec.value_type))
    value_type = expr_vec.value_type[:-1]  # drop star

    if isinstance(expr_vec, Range):
        # Dereferencing a Range yields a Slice
        return Slice(value_type, expr_vec.expr, expr_vec.size)
    else:
        # General case
        expr_vec = expr_vec.as_list()
        return List(value_type, ["*({0})".format(e) for e in expr_vec.values])


def add(x, y):
    """Add two array or singleton expressions."""
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


def concat(l, *ls):
    """Concatenate arrays."""
    if ls:
        value_type = l.value_type
        values = []

        ls = (l,) + ls
        for l in ls:
            if l.value_type != value_type:
                raise ValueError("type mismatch: '{0}' != '{1}'".format(value_type, l.value_type))
            values.extend(l.as_list().values)
        return List(value_type, values)
    else:
        return l


def _map_vec(map_name, arity, offset, iteration_index, element_index, is_facet=False):
    g_map = Singleton("int*", map_name)
    l_map = deref(add(g_map, Range("int", "{0}*{1}".format(element_index, arity), arity)))
    if isinstance(iteration_index, int):
        l_map = List(l_map.value_type, [l_map.as_list().values[iteration_index]])

    if offset is None or not any(offset):
        return l_map, None
    else:
        assert arity == len(offset)

        if isinstance(iteration_index, int):
            offset = [offset[iteration_index]]

        if not is_facet:
            return l_map, offset
        else:
            return concat(l_map, add(l_map, List("int", offset))), list(offset) * 2


def _indices(dim, map_vec, flatten=False):
    if map_vec.size == 1:
        start = "({0})*{1}".format(map_vec.as_list().values[0], dim)
        return Range(map_vec.value_type, start, dim)
    elif dim == 1:
        return map_vec
    else:
        if flatten:
            ordering = ((i, d) for d in range(dim) for i in range(map_vec.size))
        else:
            ordering = ((i, d) for i in range(map_vec.size) for d in range(dim))
        map_vec = map_vec.as_list()
        return List(map_vec.value_type,
                    [str.format("({map_item})*{dim} + {d}",
                                map_item=map_vec.values[i],
                                dim=dim, d=d)
                     for i, d in ordering])


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


def wrapper_Arg(self, args, c, col, namer, layer, is_facet=False):
    if isinstance(self.data, Mat):
        return wrapper_Mat(self, args, c, col, namer, layer, is_facet=is_facet)
    elif isinstance(self.data, Dat) and self.map is not None:
        return wrapper_DatMap(self, args, c, col, namer, layer, is_facet=is_facet)
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


def wrapper_Mat(self, args, c, col, namer, layer, is_facet=False):
    assert self.idx is not None
    assert not self._is_mixed_mat

    mat_name = args[0]
    map_names = args[1:]
    buf_name = namer('buf')

    dim = self.data.dims[0][0]  # TODO

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


def wrapper_DatMap(self, args, c, col, namer, layer, is_facet=False):
    assert len(self.data) == len(self.map)
    M = len(self.data)
    dat_names = args[:M]
    map_names = args[M:]

    buf_name = namer('vec')

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
