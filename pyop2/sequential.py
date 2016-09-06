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

from __future__ import absolute_import, print_function, division

import ctypes

from pyop2.base import ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS
from pyop2.exceptions import *
from pyop2 import host
from pyop2.mpi import collective
from pyop2.petsc_base import *
from pyop2.profiling import timed_region
from pyop2.host import Kernel, Arg  # noqa: needed by BackendSelector
from pyop2.utils import UniqueNameGenerator, cached_property
from pyop2.wrapper import DirectLayerAccess, IncrementalLayerLoop


class JITModule(host.JITModule):

    def generate_wrapper(self):
        is_facet = (self._iteration_region == ON_INTERIOR_FACETS)

        wrapper_args = []

        unique_name = UniqueNameGenerator()
        start_layer = None

        if isinstance(self._itspace._iterset, Subset):
            wrapper_args.append("int *ssinds")
        if self._itspace._extruded:
            wrapper_args += ["int nlayers"]

            region = self._iteration_region
            # Set up appropriate layer iteration bounds
            if region is ON_BOTTOM:
                start_layer = 0
                end_layer = 1
            elif region is ON_TOP:
                start_layer = "(nlayers - 1)"
                end_layer = "nlayers"
            elif region is ON_INTERIOR_FACETS:
                start_layer = 0
                end_layer = "(nlayers - 1)"
            else:
                start_layer = 0
                end_layer = "nlayers"

        def loop_body(index_name):
            kernel_args = []
            inits = []
            init_layers = []
            writebacks = []
            post_writebacks = []

            for i, arg in enumerate(self._args):
                prefix = 'arg{0}_'.format(i)
                namer = lambda sn: unique_name(prefix + sn)

                c_typenames, _1, _2 = arg.wrapper_args()

                arg_names = []
                for j, typ in enumerate(c_typenames):
                    name = namer(str(j))
                    arg_names.append(name)
                    wrapper_args.append("{typ} *{name}".format(typ=typ, name=name))  # ugh, side effect

                col_name = 'j_0' if self._itspace._extruded else None
                # layer_access = DirectLayerAccess(col_name)
                layer_access = IncrementalLayerLoop(start_layer, namer)
                arg_wrapper, kernel_arg = arg.init_and_writeback(arg_names, index_name, col_name, namer, layer_access, is_facet=is_facet)
                inits.extend(arg_wrapper.init)
                init_layers.extend(arg_wrapper.init_layer)
                writebacks.extend(arg_wrapper.writeback)
                post_writebacks.extend(arg_wrapper.post_writeback)
                kernel_args.append(kernel_arg)

            return inits, init_layers + [self._kernel.name + '(' + ', '.join(kernel_args) + ');'] + writebacks + post_writebacks

        base_inits, body3 = loop_body('i')
        if self._itspace._extruded and any(arg._is_indirect or arg._is_mat for arg in self._args):
            body2 = base_inits[:]
            body2.append("for (int {0} = {1}; {0} < {2}; {0}++) {{".format('j_0', start_layer, end_layer))
            body2.extend('\t' + line for line in body3)
            body2.append("}")
        else:
            body2 = base_inits + body3

        if isinstance(self._itspace._iterset, Subset):
            body = ["for (int {0} = start; {0} < end; {0}++) {{".format('n')]
            body.append('\t' + "int i = ssinds[n];")
            body.extend('\t' + line for line in body2)
            body.append("}")
        else:
            body = ["for (int {0} = start; {0} < end; {0}++) {{".format('i')]
            body.extend('\t' + line for line in body2)
            body.append("}")

        return ''.join(["void ", self._wrapper_name,
                        "(int start, int end, ",
                        ', '.join(wrapper_args),
                        ")\n{\n",
                        '\n'.join('\t' + line for line in body),
                        "\n}\n"])

    def set_argtypes(self, iterset, *args):
        argtypes = [ctypes.c_int, ctypes.c_int]

        if isinstance(iterset, Subset):
            argtypes.append(iterset._argtype)
        if iterset._extruded:
            argtypes.append(ctypes.c_int)

        for arg in args:
            _1, types, _2 = arg.wrapper_args()
            argtypes.extend(types)

        self._argtypes = argtypes


class ParLoop(host.ParLoop):

    def prepare_arglist(self, iterset, *args):
        arglist = []

        if isinstance(iterset, Subset):
            arglist.append(iterset._indices.ctypes.data)
        if iterset._extruded:
            arglist.append(iterset.layers - 1)

        for arg in args:
            _1, _2, values = arg.wrapper_args()
            arglist.extend(values)

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
    wrapper_fargs = ["{1} farg{0}".format(i, arg) for i, arg in enumerate(forward_args)]
    kernel_fargs = ["farg{0}".format(i) for i in xrange(len(forward_args))]

    unique_name = UniqueNameGenerator()
    wrapper_args = wrapper_fargs[:]

    def loop_body(index_name):
        kernel_args = kernel_fargs[:]
        inits = []
        writebacks = []

        for i, arg in enumerate(args):
            prefix = 'arg{0}_'.format(i)
            namer = lambda sn: unique_name(prefix + sn)

            c_typenames, _1, _2 = arg.wrapper_args()

            arg_names = []
            for j, typ in enumerate(c_typenames):
                name = namer(str(j))
                arg_names.append(name)
                wrapper_args.append("{typ} *{name}".format(typ=typ, name=name))  # ugh, side effect

            col_name = 'j_0' if itspace._extruded else None
            layer_access = DirectLayerAccess(col_name)
            arg_wrapper, kernel_arg = arg.init_and_writeback(arg_names, index_name, col_name, namer, layer_access)
            inits.extend(arg_wrapper.init)
            inits.extend(arg_wrapper.init_layer)
            writebacks.extend(arg_wrapper.writeback)
            kernel_args.append(kernel_arg)

        return inits + [kernel_name + '(' + ', '.join(kernel_args) + ');'] + writebacks

    body = loop_body('i')

    if itspace._extruded:
        body = ["int i = cell / nlayers;", "int j_0 = cell % nlayers;"] + body
        wrapper_args.append("int nlayers")
    else:
        body.insert(0, "int i = cell;")
    wrapper_args.append("int cell")

    return ''.join(["void ", wrapper_name, "(",
                    ', '.join(wrapper_args),
                    ")\n{\n",
                    '\n'.join('\t' + line for line in body),
                    "\n}\n"])


def _setup():
    pass
