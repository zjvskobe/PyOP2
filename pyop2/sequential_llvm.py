"""OP2 sequential backend, generating LLVM IR"""

from exceptions import *
from petsc_base import *

# Parallel loop API


class JITModule(base.JITModule):
    def __call__(self, *args):
        return self.translate()(*args)

    # Taking an LLVM IR kernel, create the loop wrapper
    def translate(self):
        if hasattr(self, '_fun'):
            return self._fun

        if any(arg._is_soa for arg in self._args):
            kernel_code = self._kernel.code  # TODO
        else:
            kernel_code = self._kernel.code

        code_to_compile = self.generate_wrapper()

    def generate_wrapper(self):
        # Generate loop wrapper code using llvmpy
        pass


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
