"""OP2 sequential backend, generating LLVM IR"""

from llvm.core import *
from llvm.ee import *
from llvm_cbuilder import *
from petsc_base import *
import host
import subprocess
import ctypes
import numpy as np
import llvm_cbuilder.shortnames as C


def ctype_to_llvm(dtype):
    map = {ctypes.c_long: C.int,
           ctypes.c_int: C.int,
           ctypes.c_uint: C.int,
           ctypes.c_float: C.float,
           ctypes.c_double: C.double}

    llvm_type = map.get(dtype)
    if llvm_type is None:
        raise NotImplementedError("Unsupported ctypes type %s" % (dtype))
    return llvm_type


def nptype_to_llvm(dtype):
    map = {'int16': C.int16,
           'int32': C.int32,
           'uint16': C.int16,
           'uint32': C.int32,
           'float32': C.float,
           'float64': C.double}

    llvm_type = map.get(dtype)
    if llvm_type is None:
        raise NotImplementedError("Unsupported NumPy type %s" % (dtype))
    return llvm_type


class JITModule(host.JITModule):
    def __init__(self, kernel, itspace, *args, **kwargs):
        self._kernel = kernel
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)
        self._iteration_region = kwargs.get('iterate', ALL)

    def __call__(self, *args, **kwargs):
        argtypes = kwargs.get('argtypes')
        return self.execute(argtypes, *args)

    def execute(self, argtypes, *args):
        if not hasattr(self, '_func'):
            self.translate(argtypes)

        self._func(*args)

    def translate(self, argtypes):
        llvm_module = Module.new('mod_' + self._kernel.name)

        vars = {}
        for c in Const._definitions():
            dtype = nptype_to_llvm(c.dtype.name)
            if c.cdim > 1:
                dtype = Type.array(dtype, c.cdim)

            var = llvm_module.add_global_variable(dtype, c.name)
            var.initializer = Constant.undef(dtype)
            var.global_constant = True
            vars[c.name] = var

        # Create module for kernel code, verify, and link to main wrapper
        # module.
        if self._kernel._is_llvm_kernel:
            kernel_module = Module.from_assembly(self._kernel.code)
        else:
            kernel_module = Module.from_assembly(self.c_kernel_to_llvm())

        # Ensure kernel function is externally accessible. This may not be the
        # case in the case of, for instance, a C kernel declared as "static
        # inline"
        kernel_func = kernel_module.get_function_named(self._kernel.name)
        kernel_func.linkage = LINKAGE_EXTERNAL

        kernel_module.verify()
        llvm_module.link_in(kernel_module)

        self.generate_wrapper(llvm_module, argtypes, vars)

        if configuration["debug"]:
            self._wrapper_code = str(llvm_module)

        self._dump_generated_code(str(llvm_module))

        # TODO - Handle optimisation flags depending on debug level etc

    def c_kernel_to_llvm(self):
        # TODO - Remove this before merge in to master, this is just here
        # temporarily to aid debugging (easier to write C test cases than raw
        # LLVM).

        code = """
        #include <stdbool.h>
        #include <math.h>
        """

        for c in Const._definitions():
            d = {'type': c.ctype,
                 'name': c.name,
                 'dim': c.cdim}

            if c.cdim == 1:
                code += "extern %(type)s %(name)s;\n" % d
            else:
                code += "extern %(type)s %(name)s[%(dim)s];\n" % d

        code += self._kernel.code

        if any(arg._is_soa for arg in self._args):
            code = """
            #define OP2_STRIDE(a, idx) a[idx]
            %s
            #undef OP2_STRIDE
            """ % (code)

        file = open('/tmp/kernel.c', "w")
        file.write(code)
        file.close()
        output = subprocess.check_output(['clang', '/tmp/kernel.c',
                                          '-femit-all-decls', '-O0',
                                          '-S', '-emit-llvm', '-o',
                                          '-'])
        return output

    def llvm_argtype(self, argtype):
        if issubclass(argtype, np.ctypeslib._ndptr):
            inner_type = argtype._dtype_.name
            return C.pointer(nptype_to_llvm(inner_type))
        else:
            return ctype_to_llvm(argtype)

    def generate_wrapper(self, llvm_module, argtypes, vars={}):
        _index_expr = "n"
        is_top = (self._iteration_region == ON_TOP)
        is_facet = (self._iteration_region == ON_INTERIOR_FACETS)

        llvm_argtypes = [self.llvm_argtype(type) for type in argtypes]
        functype = Type.function(C.void, llvm_argtypes)
        func = llvm_module.add_function(functype, 'wrap_' + self._kernel.name)
        cb = CBuilder(func)

        # Start/end and wrapper arguments
        start = cb.args[0]
        end = cb.args[1]
        func.args[0].name = 'start'
        func.args[1].name = 'end'
        count = 2
        for arg in self._args:
            count = arg.llvm_wrapper_arg(cb, func, vars, count)

        # Constant initialisers
        zero = Constant.int(Type.int(), 0)
        for c in Const._definitions():
            func.args[count].name = c.name + '_'
            const_var = llvm_module.get_global_variable_named(c.name)
            if c.cdim == 1:
                val = cb.builder.load(func.args[count])
                cb.builder.store(val, const_var)
            else:
                for j in range(c.cdim):
                    idx = Constant.int(Type.int(), j)
                    arg_ptr = cb.builder.gep(func.args[count], [idx])
                    const_ptr = cb.builder.gep(const_var, [zero, idx])
                    val = cb.builder.load(arg_ptr)
                    cb.builder.store(val, const_ptr)
            count += 1

        for i, arg in enumerate(self._args, 2):
            arg.llvm_wrapper_dec(cb, vars, i, is_facet=is_facet)

        for arg in self._args:
            if arg._is_vec_map:
                arg.llvm_vec_dec(cb, vars, is_facet=is_facet)


        if any(arg._is_mat for arg in self._args):
            raise NotImplementedError("Matrices not yet supported in "
                                      "sequential_llvm.")

        if self._itspace._extruded:
            raise NotImplementedError("Extruded meshes not yet supported in "
                                      "sequential_llvm")

        if any(arg._uses_itspace for arg in self._args):
            raise NotImplementedError("Arguments that use the iteration space "
                                      "(?) not yet supported in "
                                      "sequential_llvm")

        # Get a handle on our kernel function
        kernel_handle = cb.get_function_named(self._kernel.name)

        # Loop!
        one = cb.constant(C.int32, 1)
        loop_idx = cb.var_copy(start, name=_index_expr)
        with cb.loop() as loop:
            with loop.condition() as loop_cond:
                loop_cond(loop_idx < end)

            with loop.body():
                i = cb.var_copy(loop_idx, name='i')
                vars['i'] = i

                for arg in self._args:
                    if not arg._is_mat and arg._is_vec_map:
                        arg.llvm_vec_init(is_top, self._itspace.layers, cb,
                                          vars, is_facet=is_facet)

                # TODO in case of _itspace_args, use buffer decl instead
                _kernel_args = [arg.llvm_kernel_arg(cb, vars, count)
                                for count, arg in enumerate(self._args)]

                kernel_handle(*_kernel_args)

                loop_idx += one

        cb.ret()
        cb.close()
        llvm_module.verify()

        # Create CExecutor and set up params. This all needs to go in self, as
        # otherwise garbage collection may occur leading to PETSc segfaults
        self._llvm_module = llvm_module
        self._exe = CExecutor(self._llvm_module)
        self._ctype_args = argtypes
        self._func = self._exe.get_ctype_function(func, None,
                                                  *self._ctype_args)


class Arg(host.Arg):
    def llvm_arg_name(self, i=0, j=None):
        return self.c_arg_name(i, j)

    def llvm_vec_name(self):
        return self.c_arg_name() + "_vec"

    def llvm_map_name(self, i, j):
        return self.llvm_arg_name() + "_map%d_%d" % (i, j)

    def llvm_global_reduction_name(self, count=None):
        return self.c_arg_name()

    # Returns the new argnum. Kind of a hack, but needed due to how code
    # generation works...
    def llvm_wrapper_arg(self, cb, func, vars, argnum):
        if self._is_mat:
            raise NotImplementedError("Matrices not yet supported in "
                                      "sequential_llvm")
        else:
            name = self.llvm_arg_name()
            func.args[argnum].name = name
            vars[name] = cb.args[argnum]
            argnum += 1
        if self._is_indirect or self._is_mat:
            for i, map in enumerate(as_tuple(self.map, Map)):
                for j, m in enumerate(map):
                    map_name = self.llvm_map_name(i, j)
                    func.args[argnum].name = map_name
                    vars[map_name] = cb.args[argnum]
                    argnum += 1

        return argnum

    def llvm_vec_init(self, is_top, layers, cb, vars, is_facet=False):
        vec_idx = 0
        for i, (m, d) in enumerate(zip(self.map, self.data)):
            if self._flatten:
                raise NotImplementedError("Flattened maps not yet supported "
                                          "in sequential_llvm.")
            else:
                for idx in range(m.arity):
                    name = self.llvm_vec_name()
                    data = self.llvm_ind_data(vec_idx, i, cb, vars,
                                              is_top=is_top, layers=layers,
                                              offset=m.offset[idx] if is_top
                                              else None)
                    vars[name][vec_idx] = data
                    vec_idx += 1

                if is_facet:
                    raise NotImplementedError("Interior horizontal facets not "
                                              "yet supported in "
                                              "sequential_llvm")

    def llvm_vec_dec(self, cb, vars, is_facet=False):
        cdim = self.data.dataset.cdim if self._flatten else 1
        vec_name = self.llvm_vec_name()
        arity = self.map.arity * cdim * (2 if is_facet else 1)
        dtype = C.pointer(nptype_to_llvm(self.dtype.name))
        vars[vec_name] = cb.array(dtype, arity, name=vec_name)

    def llvm_wrapper_dec(self, cb, vars, argnum, is_facet=False):
        if self._is_mixed_mat or self._is_mat:
            raise NotImplementedError("Matrices not yet supported in "
                                      "sequential_llvm.")

    def llvm_kernel_arg(self, cb, vars, count, i=0, j=0, shape=(0,)):
        if self._uses_itspace:
            raise NotImplementedError("Unsupported kernel arg")
        elif self._is_indirect:
            if self._is_vec_map:
                return vars[self.llvm_vec_name()]
            return self.llvm_ind_data(self.idx, i, cb, vars)
        elif self._is_global_reduction:
            return vars[self.llvm_global_reduction_name(i)]
        elif isinstance(self.data, Global):
            return vars[self.llvm_arg_name(i)]
        else:
            # <argname> + i * <argdim>
            dim = cb.constant(C.int32, self.data.cdim)
            i_var = vars['i']
            return vars[self.llvm_arg_name()][i_var * dim].ref

    def llvm_ind_data(self, idx, i, cb, vars, j=0, is_top=False, layers=1,
                      offset=None):
        if is_top or offset:
            raise NotImplementedError("'top' arg and map offsets not yet "
                                      "supported in sequential_llvm")
        name = self.llvm_arg_name(i)
        map_name = self.llvm_map_name(i, 0)
        arity = cb.constant(C.int32, self.map.split[i].arity)
        idx_const = cb.constant(C.int32, idx)
        dim = cb.constant(C.int32, self.data[i].cdim)

        map_idx = vars['i'] * arity + idx_const
        map_data = vars[map_name][map_idx]
        arg_offs = map_data * dim
        if j != 0:
            off = cb.constant(C.int32, j)
            arg_offs += off

        ptr = cb.builder.gep(vars[name].value, [arg_offs.value])
        return CTemp(cb, ptr)


class ParLoop(host.ParLoop):
    def __init__(self, *args, **kwargs):
        host.ParLoop.__init__(self, *args, **kwargs)

    def _compute(self, part):
        fun = JITModule(self.kernel, self.it_space, *self.args,
                        direct=self.is_direct)
        if not hasattr(self, '_jit_args'):
            self._argtypes = [ctypes.c_int, ctypes.c_int]
            self._jit_args = [0, 0]
            if isinstance(self._it_space._iterset, Subset):
                raise NotImplementedError("Subsets not yet supported in "
                                          "sequential_llvm")
            for arg in self.args:
                if arg._is_mat:
                    raise NotImplementedError("Matrices not yet supported in "
                                              "sequential_llvm")
                else:
                    for d in arg.data:
                        self._argtypes.append(d._argtype)
                        self._jit_args.append(d._data)

                if arg._is_indirect or arg._is_mat:
                    maps = as_tuple(arg.map, Map)
                    for map in maps:
                        for m in map:
                            self._argtypes.append(m._argtype)
                            self._jit_args.append(m.values_with_halo)

            for c in Const._definitions():
                self._argtypes.append(c._argtype)
                self._jit_args.append(c.data)

            if len(self.offset_args) > 0:
                raise NotImplementedError("Offset args not yet supported in "
                                          "sequential_llvm")

            if self.iteration_region is not None:
                raise NotImplementedError("Iteration regions not yet "
                                          "supported in sequential_llvm")
            if self._it_space._extruded:
                raise NotImplementedError("Extruded iteration spaces not yet "
                                          "supported in sequential_llvm")

        self._jit_args[0] = part.offset
        self._jit_args[1] = part.offset + part.size
        fun(*self._jit_args, argtypes=self._argtypes, restype=None)


def _setup():
    pass
