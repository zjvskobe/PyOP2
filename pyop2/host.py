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

from textwrap import dedent

import base
from base import *
from base import _parloop_cache
from utils import as_tuple
import configuration as cfg
from find_op2 import *

class Arg(base.Arg):

    def c_arg_name(self):
        name = self.data.name
        if self._is_indirect and not (self._is_vec_map or self._uses_itspace):
            name += str(self.idx)
        return name

    def c_vec_name(self):
        return self.c_arg_name() + "_vec"

    def c_map_name(self):
        return self.c_arg_name() + "_map"

    def c_wrapper_arg(self):
        val = "PyObject *_%(name)s" % {'name' : self.c_arg_name() }
        if self._is_indirect or self._is_mat:
            val += ", PyObject *_%(name)s" % {'name' : self.c_map_name()}
            maps = as_tuple(self.map, Map)
            if len(maps) is 2:
                val += ", PyObject *_%(name)s" % {'name' : self.c_map_name()+'2'}
        return val

    def c_vec_dec(self):
        return ";\n%(type)s *%(vec_name)s[%(dim)s]" % \
               {'type' : self.ctype,
                'vec_name' : self.c_vec_name(),
                'dim' : self.map.dim}

    def c_wrapper_dec(self):
        if self._is_mat:
            val = "Mat %(name)s = (Mat)((uintptr_t)PyLong_AsUnsignedLong(_%(name)s))" % \
                 { "name": self.c_arg_name() }
        else:
            val = "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
              {'name' : self.c_arg_name(), 'type' : self.ctype}
        if self._is_indirect or self._is_mat:
            val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                   {'name' : self.c_map_name()}
        if self._is_mat:
            val += ";\nint *%(name)s2 = (int *)(((PyArrayObject *)_%(name)s2)->data)" % \
                       {'name' : self.c_map_name()}
        if self._is_vec_map:
            val += self.c_vec_dec()
        return val

    def c_ind_data(self, idx):
        return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s" % \
                {'name' : self.c_arg_name(),
                 'map_name' : self.c_map_name(),
                 'map_dim' : self.map.dim,
                 'idx' : idx,
                 'dim' : self.data.cdim}

    def c_kernel_arg_name(self):
        return "p_%s" % self.c_arg_name()

    def c_global_reduction_name(self):
        return self.c_arg_name()

    def c_tmp_name(self):
        return self.c_kernel_arg_name()

    def c_kernel_arg(self):
        if self._uses_itspace:
            if self._is_mat:
                if self.data._is_vector_field:
                    return self.c_kernel_arg_name()
                elif self.data._is_scalar_field:
                    idx = ''.join(["[i_%d]" % i for i, _ in enumerate(self.data.dims)])
                    return "(%(t)s (*)[1])&%(name)s%(idx)s" % \
                        {'t' : self.ctype,
                         'name' : self.c_kernel_arg_name(),
                         'idx' : idx}
                else:
                    raise RuntimeError("Don't know how to pass kernel arg %s" % self)
            else:
                return self.c_ind_data("i_%d" % self.idx.index)
        elif self._is_indirect:
            if self._is_vec_map:
                return self.c_vec_name()
            return self.c_ind_data(self.idx)
        elif self._is_global_reduction:
            return self.c_global_reduction_name()
        elif isinstance(self.data, Global):
            return self.c_arg_name()
        else:
            return "%(name)s + i * %(dim)s" % \
                {'name' : self.c_arg_name(),
                 'dim' : self.data.cdim}

    def c_vec_init(self):
        val = []
        for i in range(self.map._dim):
            val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                       {'vec_name' : self.c_vec_name(),
                        'idx' : i,
                        'data' : self.c_ind_data(i)} )
        return ";\n".join(val)

    def c_addto_scalar_field(self):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].dim
        ncols = maps[1].dim

        return 'addto_vector(%(mat)s, %(vals)s, %(nrows)s, %(rows)s, %(ncols)s, %(cols)s, %(insert)d)' % \
            {'mat' : self.c_arg_name(),
             'vals' : self.c_kernel_arg_name(),
             'nrows' : nrows,
             'ncols' : ncols,
             'rows' : "%s + i * %s" % (self.c_map_name(), nrows),
             'cols' : "%s2 + i * %s" % (self.c_map_name(), ncols),
             'insert' : self.access == WRITE }

    def c_addto_vector_field(self):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].dim
        ncols = maps[1].dim
        dims = self.data.sparsity.dims
        rmult = dims[0]
        cmult = dims[1]
        s = []
        for i in xrange(rmult):
            for j in xrange(cmult):
                idx = '[%d][%d]' % (i, j)
                val = "&%s%s" % (self.c_kernel_arg_name(), idx)
                row = "%(m)s * %(map)s[i * %(dim)s + i_0] + %(i)s" % \
                      {'m' : rmult,
                       'map' : self.c_map_name(),
                       'dim' : nrows,
                       'i' : i }
                col = "%(m)s * %(map)s2[i * %(dim)s + i_1] + %(j)s" % \
                      {'m' : cmult,
                       'map' : self.c_map_name(),
                       'dim' : ncols,
                       'j' : j }

                s.append('addto_scalar(%s, %s, %s, %s, %d)' \
                        % (self.c_arg_name(), val, row, col, self.access == WRITE))
        return ';\n'.join(s)

    def tmp_decl(self, extents):
        t = self.data.ctype
        if self.data._is_scalar_field:
            dims = ''.join(["[%d]" % d for d in extents])
        elif self.data._is_vector_field:
            dims = ''.join(["[%d]" % d for d in self.data.dims])
        else:
            raise RuntimeError("Don't know how to declare temp array for %s" % self)
        return "%s %s%s" % (t, self.c_tmp_name(), dims)

    def c_zero_tmp(self):
        t = self.ctype
        if self.data._is_scalar_field:
            idx = ''.join(["[i_%d]" % i for i,_ in enumerate(self.data.dims)])
            return "%(name)s%(idx)s = (%(t)s)0" % \
                {'name' : self.c_kernel_arg_name(), 't' : t, 'idx' : idx}
        elif self.data._is_vector_field:
            size = np.prod(self.data.dims)
            return "memset(%(name)s, 0, sizeof(%(t)s) * %(size)s)" % \
                {'name' : self.c_kernel_arg_name(), 't' : t, 'size' : size}
        else:
            raise RuntimeError("Don't know how to zero temp array for %s" % self)

class ParLoop(base.ParLoop):

    _cppargs = []
    _system_headers = []

    def build(self):

        key = self._cache_key
        _fun = _parloop_cache.get(key)

        if _fun is not None:
            return _fun

        from instant import inline_with_numpy

        if any(arg._is_soa for arg in self.args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            inline %(code)s
            #undef OP2_STRIDE
            """ % {'code' : self._kernel.code}
        else:
            kernel_code = """
            inline %(code)s
            """ % {'code' : self._kernel.code }
        code_to_compile = dedent(self.wrapper) % self.generate_code()

        _const_decs = '\n'.join([const._format_declaration() for const in Const._definitions()]) + '\n'

        # We need to build with mpicc since that's required by PETSc
        cc = os.environ.get('CC')
        os.environ['CC'] = 'mpicc'
        _fun = inline_with_numpy(code_to_compile, additional_declarations = kernel_code,
                                 additional_definitions = _const_decs + kernel_code,
                                 cppargs=self._cppargs + ['-O0', '-g'] if cfg.debug else [],
                                 include_dirs=[OP2_INC, get_petsc_dir()+'/include'],
                                 source_directory=os.path.dirname(os.path.abspath(__file__)),
                                 wrap_headers=["mat_utils.h"],
                                 system_headers=self._system_headers,
                                 library_dirs=[OP2_LIB, get_petsc_dir()+'/lib'],
                                 libraries=['op2_seq', 'petsc'],
                                 sources=["mat_utils.cxx"])
        if cc:
            os.environ['CC'] = cc
        else:
            os.environ.pop('CC')

        _parloop_cache[key] = _fun
        return _fun
