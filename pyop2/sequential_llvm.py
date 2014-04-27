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
           ctypes.c_float: C.float,
           ctypes.c_double: C.double}
    return map.get(dtype)


def nptype_to_llvm(dtype):
    map = {np.int16: C.int16,
           np.int32: C.int32,
           np.float32: C.float,
           np.float64: C.double}
    return map.get(dtype)


class JITModule(host.JITModule):
    def __init__(self, kernel, itspace, *args, **kwargs):
        self._kernel = kernel
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)

    def __call__(self, *args, **kwargs):
        argtypes = kwargs.get('argtypes')
        return self.execute(argtypes, *args)

    def execute(self, argtypes, *args):
        if not hasattr(self, '_func'):
            self.translate(argtypes)

        self._func(*args)

    def translate(self, argtypes):
        if any(arg._is_soa for arg in self._args):
            raise NotImplementedError("SoA arguments not yet supported in "
                                      "sequential_llvm")

        llvm_module = Module.new('mod_' + self._kernel.name)

        vars = {}
        for c in Const._definitions():
            dtype = nptype_to_llvm(c.dtype.type)
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

        # Const declarations in the C kernel -> unreferenced variable
        # -> Clang error. Add these Const items as extern variables.
        code = ""
        for c in Const._definitions():
            d = {'type': c.ctype,
                 'name': c.name,
                 'dim': c.cdim}

            if c.cdim == 1:
                code += "extern %(type)s %(name)s;\n" % d
            else:
                code += "extern %(type)s %(name)s[%(dim)s];\n" % d

        code += self._kernel.code

        file = open('/tmp/kernel.c', "w")
        file.write(code)
        file.close()
        output = subprocess.check_output(['clang', '/tmp/kernel.c', '-S',
                                          '-emit-llvm', '-o', '-'])
        return output

    def llvm_argtype(self, argtype):
        if issubclass(argtype, np.ctypeslib._ndptr):
            inner_type = argtype._dtype_.type
            return C.pointer(nptype_to_llvm(inner_type))
        else:
            return ctype_to_llvm(argtype)

    def generate_wrapper(self, llvm_module, argtypes, vars={}):
        _index_expr = "n"

        llvm_argtypes = [self.llvm_argtype(type) for type in argtypes]
        functype = Type.function(C.void, llvm_argtypes)
        func = llvm_module.add_function(functype, 'wrap_' + self._kernel.name)
        cb = CBuilder(func)

        # Start/end and wrapper arguments
        start = cb.args[0]
        end = cb.args[1]
        func.args[0].name = 'start'
        func.args[1].name = 'end'
        for i, arg in enumerate(self._args, 2):
            name = arg.llvm_arg_name()
            func.args[i].name = name
            vars[name] = cb.args[i]

        # Constant initialisers
        const_index = len(self._args) + 2
        zero = Constant.int(Type.int(), 0)
        for i, c in enumerate(Const._definitions(), const_index):
            func.args[i].name = c.name + '_'
            const_var = llvm_module.get_global_variable_named(c.name)
            if c.cdim == 1:
                val = cb.builder.load(func.args[i])
                cb.builder.store(val, const_var)
            else:
                for j in range(c.cdim):
                    idx = Constant.int(Type.int(), j)
                    arg_ptr = cb.builder.gep(func.args[i], [idx])
                    const_ptr = cb.builder.gep(const_var, [zero, idx])
                    val = cb.builder.load(arg_ptr)
                    cb.builder.store(val, const_ptr)

        # Wrapper declarations
        for i, arg in enumerate(self._args, 2):
            arg.llvm_wrapper_dec(cb, vars, i)

        if any(arg._is_global_reduction for arg in self._args):
            raise NotImplementedError("Global reductions not yet supported "
                                      "in sequential_llvm.")

        if any(arg._is_mat or arg._is_vec_map for arg in self._args):
            raise NotImplementedError("Matrices and maps not yet supported "
                                      "in sequential_llvm.")

        if self._itspace.layers > 1:
            raise NotImplementedError("Layers/extruded meshes not yet "
                                      "supported in sequential_llvm.")

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

    def llvm_wrapper_dec(self, cb, vars, argnum):
        if self._is_mixed_mat or self._is_mat or self._is_indirect or\
                self._is_vec_map:
            raise NotImplementedError("Matrices, indirection and maps not yet "
                                      "supported in sequential_llvm.")

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
                pass  # TODO
            for arg in self.args:
                for d in arg.data:
                    self._argtypes.append(d._argtype)
                    self._jit_args.append(d._data)

            if arg._is_indirect or arg._is_mat:
                pass  # TODO

            for c in Const._definitions():
                self._argtypes.append(c._argtype)
                self._jit_args.append(c.data)

        self._jit_args[0] = part.offset
        self._jit_args[1] = part.offset + part.size
        fun(*self._jit_args, argtypes=self._argtypes, restype=None)


def _setup():
    pass
