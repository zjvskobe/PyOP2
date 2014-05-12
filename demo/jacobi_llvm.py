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

# This file contains code from the original OP2 distribution, in the
# 'update' and 'res' variables. The original copyright notice follows:

# Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
# the main source directory for a full list of copyright holders.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Mike Giles may not be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."

"""PyOP2 Jacobi demo

Port of the Jacobi demo from OP2-Common.
"""

from __future__ import print_function
from pyop2 import op2, utils
import numpy as np
from math import sqrt

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('-s', '--single',
                    action='store_true',
                    help='single precision floating point mode')
parser.add_argument('-n', '--niter',
                    action='store',
                    default=2,
                    type=int,
                    help='set the number of iteration')

opt = vars(parser.parse_args())
opt["backend"] = "sequential_llvm"
op2.init(**opt)

fp_type = np.float32 if opt['single'] else np.float64

NN = 6
NITER = opt['niter']

nnode = (NN - 1) ** 2
nedge = nnode + 4 * (NN - 1) * (NN - 2)

pp = np.zeros((2 * nedge,), dtype=np.int)

A = np.zeros((nedge,), dtype=fp_type)
r = np.zeros((nnode,), dtype=fp_type)
u = np.zeros((nnode,), dtype=fp_type)
du = np.zeros((nnode,), dtype=fp_type)

e = 0

for i in xrange(1, NN):
    for j in xrange(1, NN):
        n = i - 1 + (j - 1) * (NN - 1)
        pp[2 * e] = n
        pp[2 * e + 1] = n
        A[e] = -1
        e += 1
        for p in xrange(0, 4):
            i2 = i
            j2 = j
            if p == 0:
                i2 += -1
            if p == 1:
                i2 += +1
            if p == 2:
                j2 += -1
            if p == 3:
                j2 += +1

            if i2 == 0 or i2 == NN or j2 == 0 or j2 == NN:
                r[n] += 0.25
            else:
                pp[2 * e] = n
                pp[2 * e + 1] = i2 - 1 + (j2 - 1) * (NN - 1)
                A[e] = 0.25
                e += 1


nodes = op2.Set(nnode, "nodes")
edges = op2.Set(nedge, "edges")

ppedge = op2.Map(edges, nodes, 2, pp, "ppedge")

p_A = op2.Dat(edges, data=A, name="p_A")
p_r = op2.Dat(nodes, data=r, name="p_r")
p_u = op2.Dat(nodes, data=u, name="p_u")
p_du = op2.Dat(nodes, data=du, name="p_du")

alpha = op2.Const(1, data=1.0, name="alpha", dtype=fp_type)

beta = op2.Global(1, data=1.0, name="beta", dtype=fp_type)

llvm_kernel_opt = {'llvm_kernel': True}

res = op2.Kernel("""
define void @res(double* %A, double* %u, double* %du, double* %beta) {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  store double* %A, double** %1, align 8
  store double* %u, double** %2, align 8
  store double* %du, double** %3, align 8
  store double* %beta, double** %4, align 8
  %5 = load double** %4, align 8
  %6 = load double* %5, align 8
  %7 = load double** %1, align 8
  %8 = load double* %7, align 8
  %9 = fmul double %6, %8
  %10 = load double** %2, align 8
  %11 = load double* %10, align 8
  %12 = fmul double %9, %11
  %13 = load double** %3, align 8
  %14 = load double* %13, align 8
  %15 = fadd double %14, %12
  store double %15, double* %13, align 8
  ret void
}""", "res", llvm_kernel_opt)

update = op2.Kernel("""
@alpha = external global double

; Function Attrs: nounwind uwtable
define void @update(double* %r, double* %du, double* %u, double* %u_sum, double* %u_max) {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  %5 = alloca double*, align 8
  store double* %r, double** %1, align 8
  store double* %du, double** %2, align 8
  store double* %u, double** %3, align 8
  store double* %u_sum, double** %4, align 8
  store double* %u_max, double** %5, align 8
  %6 = load double** %2, align 8
  %7 = load double* %6, align 8
  %8 = load double* @alpha, align 8
  %9 = load double** %1, align 8
  %10 = load double* %9, align 8
  %11 = fmul double %8, %10
  %12 = fadd double %7, %11
  %13 = load double** %3, align 8
  %14 = load double* %13, align 8
  %15 = fadd double %14, %12
  store double %15, double* %13, align 8
  %16 = load double** %2, align 8
  store double 0.000000e+00, double* %16, align 8
  %17 = load double** %3, align 8
  %18 = load double* %17, align 8
  %19 = load double** %3, align 8
  %20 = load double* %19, align 8
  %21 = fmul double %18, %20
  %22 = load double** %4, align 8
  %23 = load double* %22, align 8
  %24 = fadd double %23, %21
  store double %24, double* %22, align 8
  %25 = load double** %5, align 8
  %26 = load double* %25, align 8
  %27 = load double** %3, align 8
  %28 = load double* %27, align 8
  %29 = fcmp ogt double %26, %28
  br i1 %29, label %30, label %33

; <label>:30                                      ; preds = %0
  %31 = load double** %5, align 8
  %32 = load double* %31, align 8
  br label %36

; <label>:33                                      ; preds = %0
  %34 = load double** %3, align 8
  %35 = load double* %34, align 8
  br label %36

; <label>:36                                      ; preds = %33, %30
  %37 = phi double [ %32, %30 ], [ %35, %33 ]
  %38 = load double** %5, align 8
  store double %37, double* %38, align 8
  ret void
}
""", "update", llvm_kernel_opt)


for iter in xrange(0, NITER):
    op2.par_loop(res, edges,
                 p_A(op2.READ),
                 p_u(op2.READ, ppedge[1]),
                 p_du(op2.INC, ppedge[0]),
                 beta(op2.READ))
    u_sum = op2.Global(1, data=0.0, name="u_sum", dtype=fp_type)
    u_max = op2.Global(1, data=0.0, name="u_max", dtype=fp_type)

    op2.par_loop(update, nodes,
                 p_r(op2.READ),
                 p_du(op2.RW),
                 p_u(op2.INC),
                 u_sum(op2.INC),
                 u_max(op2.MAX))

    print(" u max/rms = %f %f \n" % (u_max.data[0], sqrt(u_sum.data / nnode)))


print("\nResults after %d iterations\n" % NITER)
for j in range(NN - 1, 0, -1):
    for i in range(1, NN):
        print(" %7.4f" % p_u.data[i - 1 + (j - 1) * (NN - 1)], end='')
    print("")
print("")
