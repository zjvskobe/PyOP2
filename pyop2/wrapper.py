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
