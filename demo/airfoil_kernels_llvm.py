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

# This file contains code from the original OP2 distribution, in the code
# variables. The original copyright notice follows:

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

from pyop2.op2 import Kernel

llvm_kernel_opt = {'llvm_kernel': True}

save_soln_code = """
define void @save_soln(double* %q, double* %qold) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %n = alloca i32, align 4
  store double* %q, double** %1, align 8
  store double* %qold, double** %2, align 8
  store i32 0, i32* %n, align 4
  br label %3

; <label>:3                                       ; preds = %16, %0
  %4 = load i32* %n, align 4
  %5 = icmp slt i32 %4, 4
  br i1 %5, label %6, label %19

; <label>:6                                       ; preds = %3
  %7 = load i32* %n, align 4
  %8 = sext i32 %7 to i64
  %9 = load double** %1, align 8
  %10 = getelementptr inbounds double* %9, i64 %8
  %11 = load double* %10, align 8
  %12 = load i32* %n, align 4
  %13 = sext i32 %12 to i64
  %14 = load double** %2, align 8
  %15 = getelementptr inbounds double* %14, i64 %13
  store double %11, double* %15, align 8
  br label %16

; <label>:16                                      ; preds = %6
  %17 = load i32* %n, align 4
  %18 = add nsw i32 %17, 1
  store i32 %18, i32* %n, align 4
  br label %3

; <label>:19                                      ; preds = %3
  ret void
}
"""

adt_calc_code = """
@gam = external global double
@gm1 = external global double
@cfl = external global double

define void @adt_calc(double* %x1, double* %x2, double* %x3, double* %x4, double* %q, double* %adt) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  %5 = alloca double*, align 8
  %6 = alloca double*, align 8
  %dx = alloca double, align 8
  %dy = alloca double, align 8
  %ri = alloca double, align 8
  %u = alloca double, align 8
  %v = alloca double, align 8
  %c = alloca double, align 8
  store double* %x1, double** %1, align 8
  store double* %x2, double** %2, align 8
  store double* %x3, double** %3, align 8
  store double* %x4, double** %4, align 8
  store double* %q, double** %5, align 8
  store double* %adt, double** %6, align 8
  %7 = load double** %5, align 8
  %8 = getelementptr inbounds double* %7, i64 0
  %9 = load double* %8, align 8
  %10 = fdiv double 1.000000e+00, %9
  store double %10, double* %ri, align 8
  %11 = load double* %ri, align 8
  %12 = load double** %5, align 8
  %13 = getelementptr inbounds double* %12, i64 1
  %14 = load double* %13, align 8
  %15 = fmul double %11, %14
  store double %15, double* %u, align 8
  %16 = load double* %ri, align 8
  %17 = load double** %5, align 8
  %18 = getelementptr inbounds double* %17, i64 2
  %19 = load double* %18, align 8
  %20 = fmul double %16, %19
  store double %20, double* %v, align 8
  %21 = load double* @gam, align 8
  %22 = load double* @gm1, align 8
  %23 = fmul double %21, %22
  %24 = load double* %ri, align 8
  %25 = load double** %5, align 8
  %26 = getelementptr inbounds double* %25, i64 3
  %27 = load double* %26, align 8
  %28 = fmul double %24, %27
  %29 = load double* %u, align 8
  %30 = load double* %u, align 8
  %31 = fmul double %29, %30
  %32 = load double* %v, align 8
  %33 = load double* %v, align 8
  %34 = fmul double %32, %33
  %35 = fadd double %31, %34
  %36 = fmul double 5.000000e-01, %35
  %37 = fsub double %28, %36
  %38 = fmul double %23, %37
  %39 = call double @sqrt(double %38) nounwind
  store double %39, double* %c, align 8
  %40 = load double** %2, align 8
  %41 = getelementptr inbounds double* %40, i64 0
  %42 = load double* %41, align 8
  %43 = load double** %1, align 8
  %44 = getelementptr inbounds double* %43, i64 0
  %45 = load double* %44, align 8
  %46 = fsub double %42, %45
  store double %46, double* %dx, align 8
  %47 = load double** %2, align 8
  %48 = getelementptr inbounds double* %47, i64 1
  %49 = load double* %48, align 8
  %50 = load double** %1, align 8
  %51 = getelementptr inbounds double* %50, i64 1
  %52 = load double* %51, align 8
  %53 = fsub double %49, %52
  store double %53, double* %dy, align 8
  %54 = load double* %u, align 8
  %55 = load double* %dy, align 8
  %56 = fmul double %54, %55
  %57 = load double* %v, align 8
  %58 = load double* %dx, align 8
  %59 = fmul double %57, %58
  %60 = fsub double %56, %59
  %61 = call double @fabs(double %60) nounwind readnone
  %62 = load double* %c, align 8
  %63 = load double* %dx, align 8
  %64 = load double* %dx, align 8
  %65 = fmul double %63, %64
  %66 = load double* %dy, align 8
  %67 = load double* %dy, align 8
  %68 = fmul double %66, %67
  %69 = fadd double %65, %68
  %70 = call double @sqrt(double %69) nounwind
  %71 = fmul double %62, %70
  %72 = fadd double %61, %71
  %73 = load double** %6, align 8
  store double %72, double* %73, align 8
  %74 = load double** %3, align 8
  %75 = getelementptr inbounds double* %74, i64 0
  %76 = load double* %75, align 8
  %77 = load double** %2, align 8
  %78 = getelementptr inbounds double* %77, i64 0
  %79 = load double* %78, align 8
  %80 = fsub double %76, %79
  store double %80, double* %dx, align 8
  %81 = load double** %3, align 8
  %82 = getelementptr inbounds double* %81, i64 1
  %83 = load double* %82, align 8
  %84 = load double** %2, align 8
  %85 = getelementptr inbounds double* %84, i64 1
  %86 = load double* %85, align 8
  %87 = fsub double %83, %86
  store double %87, double* %dy, align 8
  %88 = load double* %u, align 8
  %89 = load double* %dy, align 8
  %90 = fmul double %88, %89
  %91 = load double* %v, align 8
  %92 = load double* %dx, align 8
  %93 = fmul double %91, %92
  %94 = fsub double %90, %93
  %95 = call double @fabs(double %94) nounwind readnone
  %96 = load double* %c, align 8
  %97 = load double* %dx, align 8
  %98 = load double* %dx, align 8
  %99 = fmul double %97, %98
  %100 = load double* %dy, align 8
  %101 = load double* %dy, align 8
  %102 = fmul double %100, %101
  %103 = fadd double %99, %102
  %104 = call double @sqrt(double %103) nounwind
  %105 = fmul double %96, %104
  %106 = fadd double %95, %105
  %107 = load double** %6, align 8
  %108 = load double* %107, align 8
  %109 = fadd double %108, %106
  store double %109, double* %107, align 8
  %110 = load double** %4, align 8
  %111 = getelementptr inbounds double* %110, i64 0
  %112 = load double* %111, align 8
  %113 = load double** %3, align 8
  %114 = getelementptr inbounds double* %113, i64 0
  %115 = load double* %114, align 8
  %116 = fsub double %112, %115
  store double %116, double* %dx, align 8
  %117 = load double** %4, align 8
  %118 = getelementptr inbounds double* %117, i64 1
  %119 = load double* %118, align 8
  %120 = load double** %3, align 8
  %121 = getelementptr inbounds double* %120, i64 1
  %122 = load double* %121, align 8
  %123 = fsub double %119, %122
  store double %123, double* %dy, align 8
  %124 = load double* %u, align 8
  %125 = load double* %dy, align 8
  %126 = fmul double %124, %125
  %127 = load double* %v, align 8
  %128 = load double* %dx, align 8
  %129 = fmul double %127, %128
  %130 = fsub double %126, %129
  %131 = call double @fabs(double %130) nounwind readnone
  %132 = load double* %c, align 8
  %133 = load double* %dx, align 8
  %134 = load double* %dx, align 8
  %135 = fmul double %133, %134
  %136 = load double* %dy, align 8
  %137 = load double* %dy, align 8
  %138 = fmul double %136, %137
  %139 = fadd double %135, %138
  %140 = call double @sqrt(double %139) nounwind
  %141 = fmul double %132, %140
  %142 = fadd double %131, %141
  %143 = load double** %6, align 8
  %144 = load double* %143, align 8
  %145 = fadd double %144, %142
  store double %145, double* %143, align 8
  %146 = load double** %1, align 8
  %147 = getelementptr inbounds double* %146, i64 0
  %148 = load double* %147, align 8
  %149 = load double** %4, align 8
  %150 = getelementptr inbounds double* %149, i64 0
  %151 = load double* %150, align 8
  %152 = fsub double %148, %151
  store double %152, double* %dx, align 8
  %153 = load double** %1, align 8
  %154 = getelementptr inbounds double* %153, i64 1
  %155 = load double* %154, align 8
  %156 = load double** %4, align 8
  %157 = getelementptr inbounds double* %156, i64 1
  %158 = load double* %157, align 8
  %159 = fsub double %155, %158
  store double %159, double* %dy, align 8
  %160 = load double* %u, align 8
  %161 = load double* %dy, align 8
  %162 = fmul double %160, %161
  %163 = load double* %v, align 8
  %164 = load double* %dx, align 8
  %165 = fmul double %163, %164
  %166 = fsub double %162, %165
  %167 = call double @fabs(double %166) nounwind readnone
  %168 = load double* %c, align 8
  %169 = load double* %dx, align 8
  %170 = load double* %dx, align 8
  %171 = fmul double %169, %170
  %172 = load double* %dy, align 8
  %173 = load double* %dy, align 8
  %174 = fmul double %172, %173
  %175 = fadd double %171, %174
  %176 = call double @sqrt(double %175) nounwind
  %177 = fmul double %168, %176
  %178 = fadd double %167, %177
  %179 = load double** %6, align 8
  %180 = load double* %179, align 8
  %181 = fadd double %180, %178
  store double %181, double* %179, align 8
  %182 = load double** %6, align 8
  %183 = load double* %182, align 8
  %184 = load double* @cfl, align 8
  %185 = fdiv double %183, %184
  %186 = load double** %6, align 8
  store double %185, double* %186, align 8
  ret void
}

declare double @sqrt(double) nounwind

declare double @fabs(double) nounwind readnone
"""

res_calc_code = """
@gm1 = external global double
@eps = external global double

define void @res_calc(double* %x1, double* %x2, double* %q1, double* %q2, double* %adt1, double* %adt2, double* %res1, double* %res2) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  %5 = alloca double*, align 8
  %6 = alloca double*, align 8
  %7 = alloca double*, align 8
  %8 = alloca double*, align 8
  %dx = alloca double, align 8
  %dy = alloca double, align 8
  %mu = alloca double, align 8
  %ri = alloca double, align 8
  %p1 = alloca double, align 8
  %vol1 = alloca double, align 8
  %p2 = alloca double, align 8
  %vol2 = alloca double, align 8
  %f = alloca double, align 8
  store double* %x1, double** %1, align 8
  store double* %x2, double** %2, align 8
  store double* %q1, double** %3, align 8
  store double* %q2, double** %4, align 8
  store double* %adt1, double** %5, align 8
  store double* %adt2, double** %6, align 8
  store double* %res1, double** %7, align 8
  store double* %res2, double** %8, align 8
  %9 = load double** %1, align 8
  %10 = getelementptr inbounds double* %9, i64 0
  %11 = load double* %10, align 8
  %12 = load double** %2, align 8
  %13 = getelementptr inbounds double* %12, i64 0
  %14 = load double* %13, align 8
  %15 = fsub double %11, %14
  store double %15, double* %dx, align 8
  %16 = load double** %1, align 8
  %17 = getelementptr inbounds double* %16, i64 1
  %18 = load double* %17, align 8
  %19 = load double** %2, align 8
  %20 = getelementptr inbounds double* %19, i64 1
  %21 = load double* %20, align 8
  %22 = fsub double %18, %21
  store double %22, double* %dy, align 8
  %23 = load double** %3, align 8
  %24 = getelementptr inbounds double* %23, i64 0
  %25 = load double* %24, align 8
  %26 = fdiv double 1.000000e+00, %25
  store double %26, double* %ri, align 8
  %27 = load double* @gm1, align 8
  %28 = load double** %3, align 8
  %29 = getelementptr inbounds double* %28, i64 3
  %30 = load double* %29, align 8
  %31 = load double* %ri, align 8
  %32 = fmul double 5.000000e-01, %31
  %33 = load double** %3, align 8
  %34 = getelementptr inbounds double* %33, i64 1
  %35 = load double* %34, align 8
  %36 = load double** %3, align 8
  %37 = getelementptr inbounds double* %36, i64 1
  %38 = load double* %37, align 8
  %39 = fmul double %35, %38
  %40 = load double** %3, align 8
  %41 = getelementptr inbounds double* %40, i64 2
  %42 = load double* %41, align 8
  %43 = load double** %3, align 8
  %44 = getelementptr inbounds double* %43, i64 2
  %45 = load double* %44, align 8
  %46 = fmul double %42, %45
  %47 = fadd double %39, %46
  %48 = fmul double %32, %47
  %49 = fsub double %30, %48
  %50 = fmul double %27, %49
  store double %50, double* %p1, align 8
  %51 = load double* %ri, align 8
  %52 = load double** %3, align 8
  %53 = getelementptr inbounds double* %52, i64 1
  %54 = load double* %53, align 8
  %55 = load double* %dy, align 8
  %56 = fmul double %54, %55
  %57 = load double** %3, align 8
  %58 = getelementptr inbounds double* %57, i64 2
  %59 = load double* %58, align 8
  %60 = load double* %dx, align 8
  %61 = fmul double %59, %60
  %62 = fsub double %56, %61
  %63 = fmul double %51, %62
  store double %63, double* %vol1, align 8
  %64 = load double** %4, align 8
  %65 = getelementptr inbounds double* %64, i64 0
  %66 = load double* %65, align 8
  %67 = fdiv double 1.000000e+00, %66
  store double %67, double* %ri, align 8
  %68 = load double* @gm1, align 8
  %69 = load double** %4, align 8
  %70 = getelementptr inbounds double* %69, i64 3
  %71 = load double* %70, align 8
  %72 = load double* %ri, align 8
  %73 = fmul double 5.000000e-01, %72
  %74 = load double** %4, align 8
  %75 = getelementptr inbounds double* %74, i64 1
  %76 = load double* %75, align 8
  %77 = load double** %4, align 8
  %78 = getelementptr inbounds double* %77, i64 1
  %79 = load double* %78, align 8
  %80 = fmul double %76, %79
  %81 = load double** %4, align 8
  %82 = getelementptr inbounds double* %81, i64 2
  %83 = load double* %82, align 8
  %84 = load double** %4, align 8
  %85 = getelementptr inbounds double* %84, i64 2
  %86 = load double* %85, align 8
  %87 = fmul double %83, %86
  %88 = fadd double %80, %87
  %89 = fmul double %73, %88
  %90 = fsub double %71, %89
  %91 = fmul double %68, %90
  store double %91, double* %p2, align 8
  %92 = load double* %ri, align 8
  %93 = load double** %4, align 8
  %94 = getelementptr inbounds double* %93, i64 1
  %95 = load double* %94, align 8
  %96 = load double* %dy, align 8
  %97 = fmul double %95, %96
  %98 = load double** %4, align 8
  %99 = getelementptr inbounds double* %98, i64 2
  %100 = load double* %99, align 8
  %101 = load double* %dx, align 8
  %102 = fmul double %100, %101
  %103 = fsub double %97, %102
  %104 = fmul double %92, %103
  store double %104, double* %vol2, align 8
  %105 = load double** %5, align 8
  %106 = load double* %105, align 8
  %107 = load double** %6, align 8
  %108 = load double* %107, align 8
  %109 = fadd double %106, %108
  %110 = fmul double 5.000000e-01, %109
  %111 = load double* @eps, align 8
  %112 = fmul double %110, %111
  store double %112, double* %mu, align 8
  %113 = load double* %vol1, align 8
  %114 = load double** %3, align 8
  %115 = getelementptr inbounds double* %114, i64 0
  %116 = load double* %115, align 8
  %117 = fmul double %113, %116
  %118 = load double* %vol2, align 8
  %119 = load double** %4, align 8
  %120 = getelementptr inbounds double* %119, i64 0
  %121 = load double* %120, align 8
  %122 = fmul double %118, %121
  %123 = fadd double %117, %122
  %124 = fmul double 5.000000e-01, %123
  %125 = load double* %mu, align 8
  %126 = load double** %3, align 8
  %127 = getelementptr inbounds double* %126, i64 0
  %128 = load double* %127, align 8
  %129 = load double** %4, align 8
  %130 = getelementptr inbounds double* %129, i64 0
  %131 = load double* %130, align 8
  %132 = fsub double %128, %131
  %133 = fmul double %125, %132
  %134 = fadd double %124, %133
  store double %134, double* %f, align 8
  %135 = load double* %f, align 8
  %136 = load double** %7, align 8
  %137 = getelementptr inbounds double* %136, i64 0
  %138 = load double* %137, align 8
  %139 = fadd double %138, %135
  store double %139, double* %137, align 8
  %140 = load double* %f, align 8
  %141 = load double** %8, align 8
  %142 = getelementptr inbounds double* %141, i64 0
  %143 = load double* %142, align 8
  %144 = fsub double %143, %140
  store double %144, double* %142, align 8
  %145 = load double* %vol1, align 8
  %146 = load double** %3, align 8
  %147 = getelementptr inbounds double* %146, i64 1
  %148 = load double* %147, align 8
  %149 = fmul double %145, %148
  %150 = load double* %p1, align 8
  %151 = load double* %dy, align 8
  %152 = fmul double %150, %151
  %153 = fadd double %149, %152
  %154 = load double* %vol2, align 8
  %155 = load double** %4, align 8
  %156 = getelementptr inbounds double* %155, i64 1
  %157 = load double* %156, align 8
  %158 = fmul double %154, %157
  %159 = fadd double %153, %158
  %160 = load double* %p2, align 8
  %161 = load double* %dy, align 8
  %162 = fmul double %160, %161
  %163 = fadd double %159, %162
  %164 = fmul double 5.000000e-01, %163
  %165 = load double* %mu, align 8
  %166 = load double** %3, align 8
  %167 = getelementptr inbounds double* %166, i64 1
  %168 = load double* %167, align 8
  %169 = load double** %4, align 8
  %170 = getelementptr inbounds double* %169, i64 1
  %171 = load double* %170, align 8
  %172 = fsub double %168, %171
  %173 = fmul double %165, %172
  %174 = fadd double %164, %173
  store double %174, double* %f, align 8
  %175 = load double* %f, align 8
  %176 = load double** %7, align 8
  %177 = getelementptr inbounds double* %176, i64 1
  %178 = load double* %177, align 8
  %179 = fadd double %178, %175
  store double %179, double* %177, align 8
  %180 = load double* %f, align 8
  %181 = load double** %8, align 8
  %182 = getelementptr inbounds double* %181, i64 1
  %183 = load double* %182, align 8
  %184 = fsub double %183, %180
  store double %184, double* %182, align 8
  %185 = load double* %vol1, align 8
  %186 = load double** %3, align 8
  %187 = getelementptr inbounds double* %186, i64 2
  %188 = load double* %187, align 8
  %189 = fmul double %185, %188
  %190 = load double* %p1, align 8
  %191 = load double* %dx, align 8
  %192 = fmul double %190, %191
  %193 = fsub double %189, %192
  %194 = load double* %vol2, align 8
  %195 = load double** %4, align 8
  %196 = getelementptr inbounds double* %195, i64 2
  %197 = load double* %196, align 8
  %198 = fmul double %194, %197
  %199 = fadd double %193, %198
  %200 = load double* %p2, align 8
  %201 = load double* %dx, align 8
  %202 = fmul double %200, %201
  %203 = fsub double %199, %202
  %204 = fmul double 5.000000e-01, %203
  %205 = load double* %mu, align 8
  %206 = load double** %3, align 8
  %207 = getelementptr inbounds double* %206, i64 2
  %208 = load double* %207, align 8
  %209 = load double** %4, align 8
  %210 = getelementptr inbounds double* %209, i64 2
  %211 = load double* %210, align 8
  %212 = fsub double %208, %211
  %213 = fmul double %205, %212
  %214 = fadd double %204, %213
  store double %214, double* %f, align 8
  %215 = load double* %f, align 8
  %216 = load double** %7, align 8
  %217 = getelementptr inbounds double* %216, i64 2
  %218 = load double* %217, align 8
  %219 = fadd double %218, %215
  store double %219, double* %217, align 8
  %220 = load double* %f, align 8
  %221 = load double** %8, align 8
  %222 = getelementptr inbounds double* %221, i64 2
  %223 = load double* %222, align 8
  %224 = fsub double %223, %220
  store double %224, double* %222, align 8
  %225 = load double* %vol1, align 8
  %226 = load double** %3, align 8
  %227 = getelementptr inbounds double* %226, i64 3
  %228 = load double* %227, align 8
  %229 = load double* %p1, align 8
  %230 = fadd double %228, %229
  %231 = fmul double %225, %230
  %232 = load double* %vol2, align 8
  %233 = load double** %4, align 8
  %234 = getelementptr inbounds double* %233, i64 3
  %235 = load double* %234, align 8
  %236 = load double* %p2, align 8
  %237 = fadd double %235, %236
  %238 = fmul double %232, %237
  %239 = fadd double %231, %238
  %240 = fmul double 5.000000e-01, %239
  %241 = load double* %mu, align 8
  %242 = load double** %3, align 8
  %243 = getelementptr inbounds double* %242, i64 3
  %244 = load double* %243, align 8
  %245 = load double** %4, align 8
  %246 = getelementptr inbounds double* %245, i64 3
  %247 = load double* %246, align 8
  %248 = fsub double %244, %247
  %249 = fmul double %241, %248
  %250 = fadd double %240, %249
  store double %250, double* %f, align 8
  %251 = load double* %f, align 8
  %252 = load double** %7, align 8
  %253 = getelementptr inbounds double* %252, i64 3
  %254 = load double* %253, align 8
  %255 = fadd double %254, %251
  store double %255, double* %253, align 8
  %256 = load double* %f, align 8
  %257 = load double** %8, align 8
  %258 = getelementptr inbounds double* %257, i64 3
  %259 = load double* %258, align 8
  %260 = fsub double %259, %256
  store double %260, double* %258, align 8
  ret void
}
"""

bres_calc_code = """
@gm1 = external global double
@qinf = external global [4 x double]
@eps = external global double

define void @bres_calc(double* %x1, double* %x2, double* %q1, double* %adt1, double* %res1, i32* %bound) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  %5 = alloca double*, align 8
  %6 = alloca i32*, align 8
  %dx = alloca double, align 8
  %dy = alloca double, align 8
  %mu = alloca double, align 8
  %ri = alloca double, align 8
  %p1 = alloca double, align 8
  %vol1 = alloca double, align 8
  %p2 = alloca double, align 8
  %vol2 = alloca double, align 8
  %f = alloca double, align 8
  store double* %x1, double** %1, align 8
  store double* %x2, double** %2, align 8
  store double* %q1, double** %3, align 8
  store double* %adt1, double** %4, align 8
  store double* %res1, double** %5, align 8
  store i32* %bound, i32** %6, align 8
  %7 = load double** %1, align 8
  %8 = getelementptr inbounds double* %7, i64 0
  %9 = load double* %8, align 8
  %10 = load double** %2, align 8
  %11 = getelementptr inbounds double* %10, i64 0
  %12 = load double* %11, align 8
  %13 = fsub double %9, %12
  store double %13, double* %dx, align 8
  %14 = load double** %1, align 8
  %15 = getelementptr inbounds double* %14, i64 1
  %16 = load double* %15, align 8
  %17 = load double** %2, align 8
  %18 = getelementptr inbounds double* %17, i64 1
  %19 = load double* %18, align 8
  %20 = fsub double %16, %19
  store double %20, double* %dy, align 8
  %21 = load double** %3, align 8
  %22 = getelementptr inbounds double* %21, i64 0
  %23 = load double* %22, align 8
  %24 = fdiv double 1.000000e+00, %23
  store double %24, double* %ri, align 8
  %25 = load double* @gm1, align 8
  %26 = load double** %3, align 8
  %27 = getelementptr inbounds double* %26, i64 3
  %28 = load double* %27, align 8
  %29 = load double* %ri, align 8
  %30 = fmul double 5.000000e-01, %29
  %31 = load double** %3, align 8
  %32 = getelementptr inbounds double* %31, i64 1
  %33 = load double* %32, align 8
  %34 = load double** %3, align 8
  %35 = getelementptr inbounds double* %34, i64 1
  %36 = load double* %35, align 8
  %37 = fmul double %33, %36
  %38 = load double** %3, align 8
  %39 = getelementptr inbounds double* %38, i64 2
  %40 = load double* %39, align 8
  %41 = load double** %3, align 8
  %42 = getelementptr inbounds double* %41, i64 2
  %43 = load double* %42, align 8
  %44 = fmul double %40, %43
  %45 = fadd double %37, %44
  %46 = fmul double %30, %45
  %47 = fsub double %28, %46
  %48 = fmul double %25, %47
  store double %48, double* %p1, align 8
  %49 = load i32** %6, align 8
  %50 = load i32* %49, align 4
  %51 = icmp eq i32 %50, 1
  br i1 %51, label %52, label %68

; <label>:52                                      ; preds = %0
  %53 = load double* %p1, align 8
  %54 = load double* %dy, align 8
  %55 = fmul double %53, %54
  %56 = load double** %5, align 8
  %57 = getelementptr inbounds double* %56, i64 1
  %58 = load double* %57, align 8
  %59 = fadd double %58, %55
  store double %59, double* %57, align 8
  %60 = load double* %p1, align 8
  %61 = fsub double -0.000000e+00, %60
  %62 = load double* %dx, align 8
  %63 = fmul double %61, %62
  %64 = load double** %5, align 8
  %65 = getelementptr inbounds double* %64, i64 2
  %66 = load double* %65, align 8
  %67 = fadd double %66, %63
  store double %67, double* %65, align 8
  br label %223

; <label>:68                                      ; preds = %0
  %69 = load double* %ri, align 8
  %70 = load double** %3, align 8
  %71 = getelementptr inbounds double* %70, i64 1
  %72 = load double* %71, align 8
  %73 = load double* %dy, align 8
  %74 = fmul double %72, %73
  %75 = load double** %3, align 8
  %76 = getelementptr inbounds double* %75, i64 2
  %77 = load double* %76, align 8
  %78 = load double* %dx, align 8
  %79 = fmul double %77, %78
  %80 = fsub double %74, %79
  %81 = fmul double %69, %80
  store double %81, double* %vol1, align 8
  %82 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 0), align 8
  %83 = fdiv double 1.000000e+00, %82
  store double %83, double* %ri, align 8
  %84 = load double* @gm1, align 8
  %85 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 3), align 8
  %86 = load double* %ri, align 8
  %87 = fmul double 5.000000e-01, %86
  %88 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 1), align 8
  %89 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 1), align 8
  %90 = fmul double %88, %89
  %91 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 2), align 8
  %92 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 2), align 8
  %93 = fmul double %91, %92
  %94 = fadd double %90, %93
  %95 = fmul double %87, %94
  %96 = fsub double %85, %95
  %97 = fmul double %84, %96
  store double %97, double* %p2, align 8
  %98 = load double* %ri, align 8
  %99 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 1), align 8
  %100 = load double* %dy, align 8
  %101 = fmul double %99, %100
  %102 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 2), align 8
  %103 = load double* %dx, align 8
  %104 = fmul double %102, %103
  %105 = fsub double %101, %104
  %106 = fmul double %98, %105
  store double %106, double* %vol2, align 8
  %107 = load double** %4, align 8
  %108 = load double* %107, align 8
  %109 = load double* @eps, align 8
  %110 = fmul double %108, %109
  store double %110, double* %mu, align 8
  %111 = load double* %vol1, align 8
  %112 = load double** %3, align 8
  %113 = getelementptr inbounds double* %112, i64 0
  %114 = load double* %113, align 8
  %115 = fmul double %111, %114
  %116 = load double* %vol2, align 8
  %117 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 0), align 8
  %118 = fmul double %116, %117
  %119 = fadd double %115, %118
  %120 = fmul double 5.000000e-01, %119
  %121 = load double* %mu, align 8
  %122 = load double** %3, align 8
  %123 = getelementptr inbounds double* %122, i64 0
  %124 = load double* %123, align 8
  %125 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 0), align 8
  %126 = fsub double %124, %125
  %127 = fmul double %121, %126
  %128 = fadd double %120, %127
  store double %128, double* %f, align 8
  %129 = load double* %f, align 8
  %130 = load double** %5, align 8
  %131 = getelementptr inbounds double* %130, i64 0
  %132 = load double* %131, align 8
  %133 = fadd double %132, %129
  store double %133, double* %131, align 8
  %134 = load double* %vol1, align 8
  %135 = load double** %3, align 8
  %136 = getelementptr inbounds double* %135, i64 1
  %137 = load double* %136, align 8
  %138 = fmul double %134, %137
  %139 = load double* %p1, align 8
  %140 = load double* %dy, align 8
  %141 = fmul double %139, %140
  %142 = fadd double %138, %141
  %143 = load double* %vol2, align 8
  %144 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 1), align 8
  %145 = fmul double %143, %144
  %146 = fadd double %142, %145
  %147 = load double* %p2, align 8
  %148 = load double* %dy, align 8
  %149 = fmul double %147, %148
  %150 = fadd double %146, %149
  %151 = fmul double 5.000000e-01, %150
  %152 = load double* %mu, align 8
  %153 = load double** %3, align 8
  %154 = getelementptr inbounds double* %153, i64 1
  %155 = load double* %154, align 8
  %156 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 1), align 8
  %157 = fsub double %155, %156
  %158 = fmul double %152, %157
  %159 = fadd double %151, %158
  store double %159, double* %f, align 8
  %160 = load double* %f, align 8
  %161 = load double** %5, align 8
  %162 = getelementptr inbounds double* %161, i64 1
  %163 = load double* %162, align 8
  %164 = fadd double %163, %160
  store double %164, double* %162, align 8
  %165 = load double* %vol1, align 8
  %166 = load double** %3, align 8
  %167 = getelementptr inbounds double* %166, i64 2
  %168 = load double* %167, align 8
  %169 = fmul double %165, %168
  %170 = load double* %p1, align 8
  %171 = load double* %dx, align 8
  %172 = fmul double %170, %171
  %173 = fsub double %169, %172
  %174 = load double* %vol2, align 8
  %175 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 2), align 8
  %176 = fmul double %174, %175
  %177 = fadd double %173, %176
  %178 = load double* %p2, align 8
  %179 = load double* %dx, align 8
  %180 = fmul double %178, %179
  %181 = fsub double %177, %180
  %182 = fmul double 5.000000e-01, %181
  %183 = load double* %mu, align 8
  %184 = load double** %3, align 8
  %185 = getelementptr inbounds double* %184, i64 2
  %186 = load double* %185, align 8
  %187 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 2), align 8
  %188 = fsub double %186, %187
  %189 = fmul double %183, %188
  %190 = fadd double %182, %189
  store double %190, double* %f, align 8
  %191 = load double* %f, align 8
  %192 = load double** %5, align 8
  %193 = getelementptr inbounds double* %192, i64 2
  %194 = load double* %193, align 8
  %195 = fadd double %194, %191
  store double %195, double* %193, align 8
  %196 = load double* %vol1, align 8
  %197 = load double** %3, align 8
  %198 = getelementptr inbounds double* %197, i64 3
  %199 = load double* %198, align 8
  %200 = load double* %p1, align 8
  %201 = fadd double %199, %200
  %202 = fmul double %196, %201
  %203 = load double* %vol2, align 8
  %204 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 3), align 8
  %205 = load double* %p2, align 8
  %206 = fadd double %204, %205
  %207 = fmul double %203, %206
  %208 = fadd double %202, %207
  %209 = fmul double 5.000000e-01, %208
  %210 = load double* %mu, align 8
  %211 = load double** %3, align 8
  %212 = getelementptr inbounds double* %211, i64 3
  %213 = load double* %212, align 8
  %214 = load double* getelementptr inbounds ([4 x double]* @qinf, i32 0, i64 3), align 8
  %215 = fsub double %213, %214
  %216 = fmul double %210, %215
  %217 = fadd double %209, %216
  store double %217, double* %f, align 8
  %218 = load double* %f, align 8
  %219 = load double** %5, align 8
  %220 = getelementptr inbounds double* %219, i64 3
  %221 = load double* %220, align 8
  %222 = fadd double %221, %218
  store double %222, double* %220, align 8
  br label %223

; <label>:223                                     ; preds = %68, %52
  ret void
}
"""

update_code = """
define void @update(double* %qold, double* %q, double* %res, double* %adt, double* %rms) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  %5 = alloca double*, align 8
  %del = alloca double, align 8
  %adti = alloca double, align 8
  %n = alloca i32, align 4
  store double* %qold, double** %1, align 8
  store double* %q, double** %2, align 8
  store double* %res, double** %3, align 8
  store double* %adt, double** %4, align 8
  store double* %rms, double** %5, align 8
  %6 = load double** %4, align 8
  %7 = load double* %6, align 8
  %8 = fdiv double 1.000000e+00, %7
  store double %8, double* %adti, align 8
  store i32 0, i32* %n, align 4
  br label %9

; <label>:9                                       ; preds = %41, %0
  %10 = load i32* %n, align 4
  %11 = icmp slt i32 %10, 4
  br i1 %11, label %12, label %44

; <label>:12                                      ; preds = %9
  %13 = load double* %adti, align 8
  %14 = load i32* %n, align 4
  %15 = sext i32 %14 to i64
  %16 = load double** %3, align 8
  %17 = getelementptr inbounds double* %16, i64 %15
  %18 = load double* %17, align 8
  %19 = fmul double %13, %18
  store double %19, double* %del, align 8
  %20 = load i32* %n, align 4
  %21 = sext i32 %20 to i64
  %22 = load double** %1, align 8
  %23 = getelementptr inbounds double* %22, i64 %21
  %24 = load double* %23, align 8
  %25 = load double* %del, align 8
  %26 = fsub double %24, %25
  %27 = load i32* %n, align 4
  %28 = sext i32 %27 to i64
  %29 = load double** %2, align 8
  %30 = getelementptr inbounds double* %29, i64 %28
  store double %26, double* %30, align 8
  %31 = load i32* %n, align 4
  %32 = sext i32 %31 to i64
  %33 = load double** %3, align 8
  %34 = getelementptr inbounds double* %33, i64 %32
  store double 0.000000e+00, double* %34, align 8
  %35 = load double* %del, align 8
  %36 = load double* %del, align 8
  %37 = fmul double %35, %36
  %38 = load double** %5, align 8
  %39 = load double* %38, align 8
  %40 = fadd double %39, %37
  store double %40, double* %38, align 8
  br label %41

; <label>:41                                      ; preds = %12
  %42 = load i32* %n, align 4
  %43 = add nsw i32 %42, 1
  store i32 %43, i32* %n, align 4
  br label %9

; <label>:44                                      ; preds = %9
  ret void
}
"""

save_soln = Kernel(save_soln_code, "save_soln", llvm_kernel_opt)
adt_calc = Kernel(adt_calc_code, "adt_calc", llvm_kernel_opt)
res_calc = Kernel(res_calc_code, "res_calc", llvm_kernel_opt)
bres_calc = Kernel(bres_calc_code, "bres_calc", llvm_kernel_opt)
update = Kernel(update_code, "update", llvm_kernel_opt)
