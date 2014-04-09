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

# Parallel loop API


_dtype_ctype_map = {np.int16: ctypes.c_int16,
                    np.int32: ctypes.c_int32,
                    np.float32: ctypes.c_float,
                    np.float64: ctypes.c_double}

class JITModule(host.JITModule):
    def __init__(self, kernel, itspace, *args, **kwargs):
        self._kernel = kernel
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)

    def __call__(self, *args):
        return self.execute(*args)

    def execute(self, *args):
        if not hasattr(self, '_func'):
            self.translate()

        ctypes_args = [ctypes.c_int32(args[0]), ctypes.c_int32(args[1])]
        ctypes_args += [self.arg_to_ctype(arg) for arg in args[2:]]

        self._func(*ctypes_args)

    def arg_to_ctype(self, arg):
        type = _dtype_ctype_map.get(arg.dtype.type)
        if type is None:
            # TODO Improve error message
            raise NotImplementedError('Unsupported Numpy type')
        return arg.ctypes.data_as(ctypes.POINTER(type))

    # Taking an LLVM IR kernel, create the loop wrapper
    def translate(self):
        if any(arg._is_soa for arg in self._args):
            raise NotImplementedError('SoA arguments not yet supported in\
                                        sequential_llvm')

        # Create module for kernel code, verify, and link to main wrapper
        # module.
        if self._kernel._is_llvm_kernel:
            kernel_module = Module.from_assembly(self._kernel.code)
        else:
            kernel_module = Module.from_assembly(self.c_kernel_to_llvm())

        kernel_module.verify()
        llvm_module = Module.new('mod_' + self._kernel.name)
        llvm_module.link_in(kernel_module)

        self.generate_wrapper(llvm_module)

        if configuration["debug"]:
            self._wrapper_code = str(llvm_module)

        self._dump_generated_code(str(llvm_module))

        # TODO - Handle optimisation flags depending on debug level etc

    def c_kernel_to_llvm(self):
        # TODO - Remove this before merge in to master, this is just here
        # temporarily to aid debugging (easier to write C test cases than raw
        # LLVM).
        file = open('/tmp/kernel.c', "w")
        file.write(self._kernel.code)
        file.close()
        output = subprocess.check_output(['clang', '/tmp/kernel.c', '-S',
                                          '-emit-llvm', '-o', '-'])
        return output

    def generate_wrapper(self, llvm_module):
        _index_expr = "n"

        # Variables dictionary to map argument names to their corresponding
        # node in the LLVM AST.
        vars = {}

        # Create wrapper function and add it to the LLVM module.
        # For argument types, explicitly say arg0/arg1 are of type int
        # (for _start and _end of iteration space).
        argtypes = [C.int32, C.int32]
        argtypes += [arg.llvm_arg_type() for arg in self._args]
        functype = Type.function(C.void, argtypes)
        func = llvm_module.add_function(functype, 'wrap_' + self._kernel.name +
                                        '__')
        _start = func.args[0]
        _end = func.args[1]

        # Name our arguments.
        _start.name = '_start'
        _end.name = '_end'
        for i, arg in enumerate(self._args):
            func.args[i + 2].name = '_' + arg.llvm_arg_name()

        cb = CBuilder(func)

        # Wrapper declarations
        for i, arg in enumerate(self._args, 2):
            arg.llvm_wrapper_dec(cb, vars, i)

        if len(Const._defs) > 0:
            raise NotImplementedError('Constant definitions not yet supported\
                                        in sequential_llvm.')

        if any(arg._is_global_reduction for arg in self._args):
            raise NotImplementedError('Global reductions not yet supported\
                                        in sequential_llvm.')

        if any(arg._is_mat or arg._is_vec_map for arg in self._args):
            raise NotImplementedError('Matrices and maps not yet supported\
                                        in sequential_llvm.')

        if self._itspace.layers > 1:
            raise NotImplementedError('Layers/extruded meshes not yet\
                                        supported in sequential_llvm.')

        if any(arg._uses_itspace for arg in self._args):
            raise NotImplementedError('Arguments that use the iteration space\
                                        (?) not yet supported in\
                                        sequential_llvm')

        # Get a handle on our kernel function
        kernel_handle = cb.get_function_named(self._kernel.name)

        # Loop!
        one = cb.constant(C.int32, 1)
        loop_idx = cb.var_copy(_start, name=_index_expr)
        end = cb.var_copy(_end, name='end')
        with cb.loop() as loop:
            with loop.condition() as loop_cond:
                loop_cond(loop_idx < end)

            with loop.body():
                i = cb.var_copy(loop_idx, name='i')
                vars['i'] = i

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
        self._ctype_args = [arg.ctype_arg_type() for arg in self._args]
        self._func = self._exe.get_ctype_function(func, None, ctypes.c_int32,
                                                  ctypes.c_int32,
                                                  *self._ctype_args)


class Arg(host.Arg):
    _dtype_llvm_map = {np.int16: C.int16,
                       np.int32: C.int32,
                       np.float32: C.float,
                       np.float64: C.double}

    def dtype_to_llvm(self, dtype):
        type = self._dtype_llvm_map.get(dtype.type)
        if type is None:
            raise NotImplementedError("Numpy type '%s' not yet supported in sequential_llvm." % dtype)
        return type

    def dtype_to_ctype(self, dtype):
        type = _dtype_ctype_map.get(dtype.type)
        if type is None:
            raise NotImplementedError("Numpy type '%s' not yet supported in sequential_llvm." % dtype)
        return type

    def llvm_arg_type(self):
        # TODO - these may not necessarily always be pointers!
        return C.pointer(self.dtype_to_llvm(self.dtype))

    def ctype_arg_type(self):
        return ctypes.POINTER(self.dtype_to_ctype(self.dtype))

    def llvm_all_arg_types(self):
        data_type = self.llvm_arg_type()
        if self._is_mat or self._is_indirect:
            raise NotImplementedError("Matrices and indirection not yet\
                                        supported in sequential_llvm.")
        else:
            return [data_type for i in range(len(self.data))]

    def llvm_arg_name(self, i=0, j=None):
        return self.c_arg_name(i, j)

    def llvm_wrapper_dec(self, cb, vars, argnum):
        if self._is_mixed_mat or self._is_mat or self._is_indirect or\
                self._is_vec_map:
            raise NotImplementedError("Matrices, indirection and maps not yet\
                                        supported in sequential_llvm.")
        else:
            # TODO handle multiple Dat's in self.data (mixed mats)
            llvm_var = cb.var_copy(cb.args[argnum], name=self.llvm_arg_name())
            vars[self.llvm_arg_name()] = llvm_var
            return llvm_var

    def llvm_kernel_arg(self, cb, vars, i=0, j=0, shape=(0,)):
        if self._uses_itspace or self._is_indirect\
                or self._is_global_reduction:
            raise NotImplementedError("Unsupported kernel arg")

        if isinstance(self.data, Global):
            return vars[self.llvm_arg_name(i)]
        else:
            # <argname> + i * <argdim>
            dim = cb.constant(C.int32, self.data.cdim)
            i = vars['i']
            return vars[self.llvm_arg_name()][i * dim].ref

# Based on ParLoop in sequential.py


class ParLoop(host.ParLoop):
    def __init__(self, *args, **kwargs):
        host.ParLoop.__init__(self, *args, **kwargs)

    def _compute(self, part):
        fun = JITModule(self.kernel, self.it_space, *self.args,
                        direct=self.is_direct)
        if not hasattr(self, '_jit_args'):
            self._jit_args = [0, 0]
            if isinstance(self._it_space._iterset, Subset):
                pass  # TODO
            for arg in self.args:
                for d in arg.data:
                    self._jit_args.append(d._data)

            if arg._is_indirect or arg._is_mat:
                pass  # TODO

        for c in Const._definitions():
            pass  # TODO

        #TODO extend with offset/layer args

        if part.size > 0:
            self._jit_args[0] = part.offset
            self._jit_args[1] = part.offset + part.size
            fun(*self._jit_args)


def _setup():
    pass
