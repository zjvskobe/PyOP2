# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012-2016, Imperial College London and
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

"""OP2 sequential backend."""

import ctypes
import itertools

from base import ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS
from exceptions import *
import host
from mpi import collective
from petsc_base import *
from profiling import timed_region
from host import Kernel, Arg  # noqa: needed by BackendSelector
from utils import cached_property


def _alternative_names(name):
    yield name
    if len(name) >= 1 and name[-1] == '_':
        name_underscore = name
    else:
        name_underscore = name + '_'
        yield name_underscore
    for i in itertools.count(1):
        yield name_underscore + str(i)


def alternative_names(name):
    """Given a suggested name, generates alternative names to avoid name
    collisions."""
    return itertools.ifilter(lambda s: s not in ['', '_'],
                             _alternative_names(name))


class UniqueNameGenerator(object):
    def __init__(self):
        self.names = set()

    def __call__(self, name):
        for alt_name in alternative_names(name):
            if alt_name not in self.names:
                self.names.add(alt_name)
                return alt_name


class JITModule(host.JITModule):

    def generate_wrapper(self):
        wrapper_args = []

        unique_name = UniqueNameGenerator()

        def loop_body(index_name):
            kernel_args = []
            inits = []
            writebacks = []

            for i, arg in enumerate(self._args):
                prefix = 'arg{0}_'.format(i)
                namer = lambda sn: unique_name(prefix + sn)

                c_typenames, _1, _2 = arg.wrapper_args()

                arg_names = []
                for j, typ in enumerate(c_typenames):
                    name = namer(str(j))
                    arg_names.append(name)
                    wrapper_args.append("{typ} *{name}".format(typ=typ, name=name))  # ugh, side effect

                init, writeback, kernel_arg = arg.init_and_writeback(arg_names, index_name, namer)
                inits.extend(init)
                writebacks.extend(writeback)
                kernel_args.append(kernel_arg)

            return inits + [self._kernel.name + '(' + ', '.join(kernel_args) + ');'] + writebacks

        if isinstance(self._itspace._iterset, Subset):
            wrapper_args.append("int *ssinds")
        if self._itspace._extruded:
            wrapper_args += ["int start_layer", "int end_layer", "int top_layer"]

        if isinstance(self._itspace._iterset, Subset):
            body = ["for (int {0} = start; {0} < end; {0}++) {{".format('n')]
            body.append('\t' + "int i = ssinds[n];")
            body.extend('\t' + line for line in loop_body('i'))
            body.append("}")
        else:
            body = ["for (int {0} = start; {0} < end; {0}++) {{".format('i')]
            body.extend('\t' + line for line in loop_body('i'))
            body.append("}")

        return ''.join(["void ", self._wrapper_name,
                        "(int start, int end, ",
                        ', '.join(wrapper_args),
                        ")\n{\n",
                        '\n'.join('\t' + line for line in body),
                        "}\n"])

    def set_argtypes(self, iterset, *args):
        argtypes = [ctypes.c_int, ctypes.c_int]

        if isinstance(iterset, Subset):
            argtypes.append(iterset._argtype)
        if iterset._extruded:
            argtypes.append(ctypes.c_int)
            argtypes.append(ctypes.c_int)
            argtypes.append(ctypes.c_int)

        for arg in args:
            _1, types, _2 = arg.wrapper_args()
            argtypes.extend(types)

        for c in Const._definitions():
            argtypes.append(c._argtype)

        self._argtypes = argtypes


class ParLoop(host.ParLoop):

    def prepare_arglist(self, iterset, *args):
        arglist = []

        if isinstance(iterset, Subset):
            arglist.append(iterset._indices.ctypes.data)
        if iterset._extruded:
            region = self.iteration_region
            # Set up appropriate layer iteration bounds
            if region is ON_BOTTOM:
                arglist.append(0)
                arglist.append(1)
                arglist.append(iterset.layers - 1)
            elif region is ON_TOP:
                arglist.append(iterset.layers - 2)
                arglist.append(iterset.layers - 1)
                arglist.append(iterset.layers - 1)
            elif region is ON_INTERIOR_FACETS:
                arglist.append(0)
                arglist.append(iterset.layers - 2)
                arglist.append(iterset.layers - 2)
            else:
                arglist.append(0)
                arglist.append(iterset.layers - 1)
                arglist.append(iterset.layers - 1)

        for arg in args:
            _1, _2, values = arg.wrapper_args()
            arglist.extend(values)

        for c in Const._definitions():
            arglist.append(c._data.ctypes.data)

        return arglist

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.it_space, *self.args,
                         direct=self.is_direct, iterate=self.iteration_region)

    @collective
    def _compute(self, part, fun, *arglist):
        with timed_region("ParLoopCKernel"):
            fun(part.offset, part.offset + part.size, *arglist)
            self.log_flops()


def generate_cell_wrapper(itspace, args, forward_args=(), kernel_name=None, wrapper_name=None):
    raise NotImplementedError("How to generate cell wrapper?")


def _setup():
    pass
