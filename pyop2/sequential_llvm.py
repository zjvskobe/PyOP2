"""OP2 sequential backend, generating LLVM IR"""

from llvm.core import *
from llvm.ee import *
from llvm_cbuilder import *
import numpy as np
import llvm_cbuilder.shortnames as C

# Parallel loop API


class JITModule(base.JITModule):
    def __init__(self, kernel, itspace, *args, **kwargs):
        self._kernel = kernel
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)
        self._translated = False

    def __call__(self, *args):
        return self.execute(*args)

    def execute(*args):
        if not self._translated:
            self.translate()

        mod = self._llvm_module

        # TODO Invoke execution engine using this LLVM module. Or maybe
        # CExecutor?


    # Taking an LLVM IR kernel, create the loop wrapper
    def translate(self):
        # Create LLVM module
        llvm_module = self._llvm_module = Module.new('mod_' +
                self._kernel.name)

        kernel_code = self._kernel.code # TODO handle soa
        code_to_compile = self.generate_wrapper(llvm_module)

    def generate_wrapper(self, llvm_module):
        _ssinds_arg = ""
        _ssinds_dec = ""
        _index_expr = "n"

        # Create wrapper function and add it to the LLVM module.
        argtypes = [type for type in C.pointer(arg.llvm_arg_type())
                    for arg in self._args]
        functype = Type.function(C.void, argtypes)
        func = llvm_module.add_function(functype, self._kernel.name +
                                        '_wrapper')

        _wrapper_decs = [arg.llvm_wrapper_dec(cb, i)
                         for arg, i in enumerate(self._args, 0)]






class Arg(base.Arg):
    _dtype_llvm_map = {np.int16: C.int16,
                       np.int32: C.int32}

    def dtype_to_llvm(self, dtype):
        type = _ctype_llvm_map.get(dtype)
        if type is None:
            raise NotImplementedError("Numpy type '%s' not yet supported in\
                                        sequential_llvm." % dtype)
        return type

    def llvm_arg_type(self):
        return self.dtype_to_llvm(self.dtype)

    def llvm_all_arg_types(self):
        data_type = self.llvm_arg_type()
        if self._is_mat or self._is_indirect:
            raise NotImplementedError("Matrices and indirection not yet\
                                        supported in sequential_llvm.")
        else:
            return [data_type for i in range(len(self.data))]

    def llvm_arg_name(self, i=0, j=None):
        return self.c_arg_name(i, j)

    def llvm_wrapper_dec(self, cb, argnum):
        if self._is_mixed_mat or self._is_mat or self._is_indrect or\
                self._is_vec_map:
            raise NotImplementedError("Matrices, indirection and maps not yet\
                                        supported in sequential_llvm.")
        else:  # TODO consider multiple args in the arg?
            return [cb.var(self.llvm_arg_type(), cb.args[argnum],
                    name=self.llvm_arg_name(i))]
                    #for i, _ in enumerate(self.data)]



# Based on ParLoop in sequential.py
class ParLoop(base.ParLoop):
    def __init__(self, *args, **kwargs):
        base.ParLoop.__init__(self, *args, **kwargs)

    def _compute(self, part):
        fun = JITModule(self.kernel, self.it_space, *self.args,
                        direct=self.is_direct)
        if not hasattr(self, '_jit_args'):
            self._jit_args = [0, 0]
            if isinstance(self._it_space._iterset, Subset):
                pass  # TODO
            else:
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
