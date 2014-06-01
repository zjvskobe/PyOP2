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

res_calc_code = """
@Ng2_xi = external global [32 x double]
@wtg2 = external global [4 x double]
@gm1 = external global double
@m2 = external global double
@gm1i = external global double

define void @res_calc(double** %x, double** %phim, double* %K, double** %res) nounwind uwtable {
  %1 = alloca double**, align 8
  %2 = alloca double**, align 8
  %3 = alloca double*, align 8
  %4 = alloca double**, align 8
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %i = alloca i32, align 4
  %det_x_xi = alloca double, align 8
  %N_x = alloca [8 x double], align 16
  %a = alloca double, align 8
  %m = alloca i32, align 4
  %m1 = alloca i32, align 4
  %m2 = alloca i32, align 4
  %m3 = alloca i32, align 4
  %m4 = alloca i32, align 4
  %m5 = alloca i32, align 4
  %b = alloca double, align 8
  %m6 = alloca i32, align 4
  %m7 = alloca i32, align 4
  %j8 = alloca i32, align 4
  %wt1 = alloca double, align 8
  %u = alloca [2 x double], align 16
  %j9 = alloca i32, align 4
  %Dk = alloca double, align 8
  %rho = alloca double, align 8
  %rc2 = alloca double, align 8
  %j10 = alloca i32, align 4
  %j11 = alloca i32, align 4
  %k12 = alloca i32, align 4
  store double** %x, double*** %1, align 8
  store double** %phim, double*** %2, align 8
  store double* %K, double** %3, align 8
  store double** %res, double*** %4, align 8
  store i32 0, i32* %j, align 4
  br label %5

; <label>:5                                       ; preds = %24, %0
  %6 = load i32* %j, align 4
  %7 = icmp slt i32 %6, 4
  br i1 %7, label %8, label %27

; <label>:8                                       ; preds = %5
  store i32 0, i32* %k, align 4
  br label %9

; <label>:9                                       ; preds = %20, %8
  %10 = load i32* %k, align 4
  %11 = icmp slt i32 %10, 4
  br i1 %11, label %12, label %23

; <label>:12                                      ; preds = %9
  %13 = load i32* %j, align 4
  %14 = mul nsw i32 %13, 4
  %15 = load i32* %k, align 4
  %16 = add nsw i32 %14, %15
  %17 = sext i32 %16 to i64
  %18 = load double** %3, align 8
  %19 = getelementptr inbounds double* %18, i64 %17
  store double 0.000000e+00, double* %19, align 8
  br label %20

; <label>:20                                      ; preds = %12
  %21 = load i32* %k, align 4
  %22 = add nsw i32 %21, 1
  store i32 %22, i32* %k, align 4
  br label %9

; <label>:23                                      ; preds = %9
  br label %24

; <label>:24                                      ; preds = %23
  %25 = load i32* %j, align 4
  %26 = add nsw i32 %25, 1
  store i32 %26, i32* %j, align 4
  br label %5

; <label>:27                                      ; preds = %5
  store i32 0, i32* %i, align 4
  br label %28

; <label>:28                                      ; preds = %437, %27
  %29 = load i32* %i, align 4
  %30 = icmp slt i32 %29, 4
  br i1 %30, label %31, label %440

; <label>:31                                      ; preds = %28
  store double 0.000000e+00, double* %det_x_xi, align 8
  store double 0.000000e+00, double* %a, align 8
  store i32 0, i32* %m, align 4
  br label %32

; <label>:32                                      ; preds = %54, %31
  %33 = load i32* %m, align 4
  %34 = icmp slt i32 %33, 4
  br i1 %34, label %35, label %57

; <label>:35                                      ; preds = %32
  %36 = load i32* %i, align 4
  %37 = mul nsw i32 4, %36
  %38 = add nsw i32 %37, 16
  %39 = load i32* %m, align 4
  %40 = add nsw i32 %38, %39
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds [32 x double]* @Ng2_xi, i32 0, i64 %41
  %43 = load double* %42, align 8
  %44 = load i32* %m, align 4
  %45 = sext i32 %44 to i64
  %46 = load double*** %1, align 8
  %47 = getelementptr inbounds double** %46, i64 %45
  %48 = load double** %47, align 8
  %49 = getelementptr inbounds double* %48, i64 1
  %50 = load double* %49, align 8
  %51 = fmul double %43, %50
  %52 = load double* %det_x_xi, align 8
  %53 = fadd double %52, %51
  store double %53, double* %det_x_xi, align 8
  br label %54

; <label>:54                                      ; preds = %35
  %55 = load i32* %m, align 4
  %56 = add nsw i32 %55, 1
  store i32 %56, i32* %m, align 4
  br label %32

; <label>:57                                      ; preds = %32
  store i32 0, i32* %m1, align 4
  br label %58

; <label>:58                                      ; preds = %74, %57
  %59 = load i32* %m1, align 4
  %60 = icmp slt i32 %59, 4
  br i1 %60, label %61, label %77

; <label>:61                                      ; preds = %58
  %62 = load double* %det_x_xi, align 8
  %63 = load i32* %i, align 4
  %64 = mul nsw i32 4, %63
  %65 = load i32* %m1, align 4
  %66 = add nsw i32 %64, %65
  %67 = sext i32 %66 to i64
  %68 = getelementptr inbounds [32 x double]* @Ng2_xi, i32 0, i64 %67
  %69 = load double* %68, align 8
  %70 = fmul double %62, %69
  %71 = load i32* %m1, align 4
  %72 = sext i32 %71 to i64
  %73 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %72
  store double %70, double* %73, align 8
  br label %74

; <label>:74                                      ; preds = %61
  %75 = load i32* %m1, align 4
  %76 = add nsw i32 %75, 1
  store i32 %76, i32* %m1, align 4
  br label %58

; <label>:77                                      ; preds = %58
  store double 0.000000e+00, double* %a, align 8
  store i32 0, i32* %m2, align 4
  br label %78

; <label>:78                                      ; preds = %99, %77
  %79 = load i32* %m2, align 4
  %80 = icmp slt i32 %79, 4
  br i1 %80, label %81, label %102

; <label>:81                                      ; preds = %78
  %82 = load i32* %i, align 4
  %83 = mul nsw i32 4, %82
  %84 = load i32* %m2, align 4
  %85 = add nsw i32 %83, %84
  %86 = sext i32 %85 to i64
  %87 = getelementptr inbounds [32 x double]* @Ng2_xi, i32 0, i64 %86
  %88 = load double* %87, align 8
  %89 = load i32* %m2, align 4
  %90 = sext i32 %89 to i64
  %91 = load double*** %1, align 8
  %92 = getelementptr inbounds double** %91, i64 %90
  %93 = load double** %92, align 8
  %94 = getelementptr inbounds double* %93, i64 0
  %95 = load double* %94, align 8
  %96 = fmul double %88, %95
  %97 = load double* %a, align 8
  %98 = fadd double %97, %96
  store double %98, double* %a, align 8
  br label %99

; <label>:99                                      ; preds = %81
  %100 = load i32* %m2, align 4
  %101 = add nsw i32 %100, 1
  store i32 %101, i32* %m2, align 4
  br label %78

; <label>:102                                     ; preds = %78
  store i32 0, i32* %m3, align 4
  br label %103

; <label>:103                                     ; preds = %121, %102
  %104 = load i32* %m3, align 4
  %105 = icmp slt i32 %104, 4
  br i1 %105, label %106, label %124

; <label>:106                                     ; preds = %103
  %107 = load double* %a, align 8
  %108 = load i32* %i, align 4
  %109 = mul nsw i32 4, %108
  %110 = add nsw i32 %109, 16
  %111 = load i32* %m3, align 4
  %112 = add nsw i32 %110, %111
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds [32 x double]* @Ng2_xi, i32 0, i64 %113
  %115 = load double* %114, align 8
  %116 = fmul double %107, %115
  %117 = load i32* %m3, align 4
  %118 = add nsw i32 4, %117
  %119 = sext i32 %118 to i64
  %120 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %119
  store double %116, double* %120, align 8
  br label %121

; <label>:121                                     ; preds = %106
  %122 = load i32* %m3, align 4
  %123 = add nsw i32 %122, 1
  store i32 %123, i32* %m3, align 4
  br label %103

; <label>:124                                     ; preds = %103
  %125 = load double* %a, align 8
  %126 = load double* %det_x_xi, align 8
  %127 = fmul double %126, %125
  store double %127, double* %det_x_xi, align 8
  store double 0.000000e+00, double* %a, align 8
  store i32 0, i32* %m4, align 4
  br label %128

; <label>:128                                     ; preds = %149, %124
  %129 = load i32* %m4, align 4
  %130 = icmp slt i32 %129, 4
  br i1 %130, label %131, label %152

; <label>:131                                     ; preds = %128
  %132 = load i32* %i, align 4
  %133 = mul nsw i32 4, %132
  %134 = load i32* %m4, align 4
  %135 = add nsw i32 %133, %134
  %136 = sext i32 %135 to i64
  %137 = getelementptr inbounds [32 x double]* @Ng2_xi, i32 0, i64 %136
  %138 = load double* %137, align 8
  %139 = load i32* %m4, align 4
  %140 = sext i32 %139 to i64
  %141 = load double*** %1, align 8
  %142 = getelementptr inbounds double** %141, i64 %140
  %143 = load double** %142, align 8
  %144 = getelementptr inbounds double* %143, i64 1
  %145 = load double* %144, align 8
  %146 = fmul double %138, %145
  %147 = load double* %a, align 8
  %148 = fadd double %147, %146
  store double %148, double* %a, align 8
  br label %149

; <label>:149                                     ; preds = %131
  %150 = load i32* %m4, align 4
  %151 = add nsw i32 %150, 1
  store i32 %151, i32* %m4, align 4
  br label %128

; <label>:152                                     ; preds = %128
  store i32 0, i32* %m5, align 4
  br label %153

; <label>:153                                     ; preds = %172, %152
  %154 = load i32* %m5, align 4
  %155 = icmp slt i32 %154, 4
  br i1 %155, label %156, label %175

; <label>:156                                     ; preds = %153
  %157 = load double* %a, align 8
  %158 = load i32* %i, align 4
  %159 = mul nsw i32 4, %158
  %160 = add nsw i32 %159, 16
  %161 = load i32* %m5, align 4
  %162 = add nsw i32 %160, %161
  %163 = sext i32 %162 to i64
  %164 = getelementptr inbounds [32 x double]* @Ng2_xi, i32 0, i64 %163
  %165 = load double* %164, align 8
  %166 = fmul double %157, %165
  %167 = load i32* %m5, align 4
  %168 = sext i32 %167 to i64
  %169 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %168
  %170 = load double* %169, align 8
  %171 = fsub double %170, %166
  store double %171, double* %169, align 8
  br label %172

; <label>:172                                     ; preds = %156
  %173 = load i32* %m5, align 4
  %174 = add nsw i32 %173, 1
  store i32 %174, i32* %m5, align 4
  br label %153

; <label>:175                                     ; preds = %153
  store double 0.000000e+00, double* %b, align 8
  store i32 0, i32* %m6, align 4
  br label %176

; <label>:176                                     ; preds = %198, %175
  %177 = load i32* %m6, align 4
  %178 = icmp slt i32 %177, 4
  br i1 %178, label %179, label %201

; <label>:179                                     ; preds = %176
  %180 = load i32* %i, align 4
  %181 = mul nsw i32 4, %180
  %182 = add nsw i32 %181, 16
  %183 = load i32* %m6, align 4
  %184 = add nsw i32 %182, %183
  %185 = sext i32 %184 to i64
  %186 = getelementptr inbounds [32 x double]* @Ng2_xi, i32 0, i64 %185
  %187 = load double* %186, align 8
  %188 = load i32* %m6, align 4
  %189 = sext i32 %188 to i64
  %190 = load double*** %1, align 8
  %191 = getelementptr inbounds double** %190, i64 %189
  %192 = load double** %191, align 8
  %193 = getelementptr inbounds double* %192, i64 0
  %194 = load double* %193, align 8
  %195 = fmul double %187, %194
  %196 = load double* %b, align 8
  %197 = fadd double %196, %195
  store double %197, double* %b, align 8
  br label %198

; <label>:198                                     ; preds = %179
  %199 = load i32* %m6, align 4
  %200 = add nsw i32 %199, 1
  store i32 %200, i32* %m6, align 4
  br label %176

; <label>:201                                     ; preds = %176
  store i32 0, i32* %m7, align 4
  br label %202

; <label>:202                                     ; preds = %221, %201
  %203 = load i32* %m7, align 4
  %204 = icmp slt i32 %203, 4
  br i1 %204, label %205, label %224

; <label>:205                                     ; preds = %202
  %206 = load double* %b, align 8
  %207 = load i32* %i, align 4
  %208 = mul nsw i32 4, %207
  %209 = load i32* %m7, align 4
  %210 = add nsw i32 %208, %209
  %211 = sext i32 %210 to i64
  %212 = getelementptr inbounds [32 x double]* @Ng2_xi, i32 0, i64 %211
  %213 = load double* %212, align 8
  %214 = fmul double %206, %213
  %215 = load i32* %m7, align 4
  %216 = add nsw i32 4, %215
  %217 = sext i32 %216 to i64
  %218 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %217
  %219 = load double* %218, align 8
  %220 = fsub double %219, %214
  store double %220, double* %218, align 8
  br label %221

; <label>:221                                     ; preds = %205
  %222 = load i32* %m7, align 4
  %223 = add nsw i32 %222, 1
  store i32 %223, i32* %m7, align 4
  br label %202

; <label>:224                                     ; preds = %202
  %225 = load double* %a, align 8
  %226 = load double* %b, align 8
  %227 = fmul double %225, %226
  %228 = load double* %det_x_xi, align 8
  %229 = fsub double %228, %227
  store double %229, double* %det_x_xi, align 8
  store i32 0, i32* %j8, align 4
  br label %230

; <label>:230                                     ; preds = %240, %224
  %231 = load i32* %j8, align 4
  %232 = icmp slt i32 %231, 8
  br i1 %232, label %233, label %243

; <label>:233                                     ; preds = %230
  %234 = load double* %det_x_xi, align 8
  %235 = load i32* %j8, align 4
  %236 = sext i32 %235 to i64
  %237 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %236
  %238 = load double* %237, align 8
  %239 = fdiv double %238, %234
  store double %239, double* %237, align 8
  br label %240

; <label>:240                                     ; preds = %233
  %241 = load i32* %j8, align 4
  %242 = add nsw i32 %241, 1
  store i32 %242, i32* %j8, align 4
  br label %230

; <label>:243                                     ; preds = %230
  %244 = load i32* %i, align 4
  %245 = sext i32 %244 to i64
  %246 = getelementptr inbounds [4 x double]* @wtg2, i32 0, i64 %245
  %247 = load double* %246, align 8
  %248 = load double* %det_x_xi, align 8
  %249 = fmul double %247, %248
  store double %249, double* %wt1, align 8
  %250 = bitcast [2 x double]* %u to i8*
  call void @llvm.memset.p0i8.i64(i8* %250, i8 0, i64 16, i32 16, i1 false)
  store i32 0, i32* %j9, align 4
  br label %251

; <label>:251                                     ; preds = %286, %243
  %252 = load i32* %j9, align 4
  %253 = icmp slt i32 %252, 4
  br i1 %253, label %254, label %289

; <label>:254                                     ; preds = %251
  %255 = load i32* %j9, align 4
  %256 = sext i32 %255 to i64
  %257 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %256
  %258 = load double* %257, align 8
  %259 = load i32* %j9, align 4
  %260 = sext i32 %259 to i64
  %261 = load double*** %2, align 8
  %262 = getelementptr inbounds double** %261, i64 %260
  %263 = load double** %262, align 8
  %264 = getelementptr inbounds double* %263, i64 0
  %265 = load double* %264, align 8
  %266 = fmul double %258, %265
  %267 = getelementptr inbounds [2 x double]* %u, i32 0, i64 0
  %268 = load double* %267, align 8
  %269 = fadd double %268, %266
  store double %269, double* %267, align 8
  %270 = load i32* %j9, align 4
  %271 = add nsw i32 4, %270
  %272 = sext i32 %271 to i64
  %273 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %272
  %274 = load double* %273, align 8
  %275 = load i32* %j9, align 4
  %276 = sext i32 %275 to i64
  %277 = load double*** %2, align 8
  %278 = getelementptr inbounds double** %277, i64 %276
  %279 = load double** %278, align 8
  %280 = getelementptr inbounds double* %279, i64 0
  %281 = load double* %280, align 8
  %282 = fmul double %274, %281
  %283 = getelementptr inbounds [2 x double]* %u, i32 0, i64 1
  %284 = load double* %283, align 8
  %285 = fadd double %284, %282
  store double %285, double* %283, align 8
  br label %286

; <label>:286                                     ; preds = %254
  %287 = load i32* %j9, align 4
  %288 = add nsw i32 %287, 1
  store i32 %288, i32* %j9, align 4
  br label %251

; <label>:289                                     ; preds = %251
  %290 = load double* @gm1, align 8
  %291 = fmul double 5.000000e-01, %290
  %292 = load double* @m2, align 8
  %293 = getelementptr inbounds [2 x double]* %u, i32 0, i64 0
  %294 = load double* %293, align 8
  %295 = getelementptr inbounds [2 x double]* %u, i32 0, i64 0
  %296 = load double* %295, align 8
  %297 = fmul double %294, %296
  %298 = getelementptr inbounds [2 x double]* %u, i32 0, i64 1
  %299 = load double* %298, align 8
  %300 = getelementptr inbounds [2 x double]* %u, i32 0, i64 1
  %301 = load double* %300, align 8
  %302 = fmul double %299, %301
  %303 = fadd double %297, %302
  %304 = fsub double %292, %303
  %305 = fmul double %291, %304
  %306 = fadd double 1.000000e+00, %305
  store double %306, double* %Dk, align 8
  %307 = load double* %Dk, align 8
  %308 = load double* @gm1i, align 8
  %309 = call double @pow(double %307, double %308) nounwind
  store double %309, double* %rho, align 8
  %310 = load double* %rho, align 8
  %311 = load double* %Dk, align 8
  %312 = fdiv double %310, %311
  store double %312, double* %rc2, align 8
  store i32 0, i32* %j10, align 4
  br label %313

; <label>:313                                     ; preds = %345, %289
  %314 = load i32* %j10, align 4
  %315 = icmp slt i32 %314, 4
  br i1 %315, label %316, label %348

; <label>:316                                     ; preds = %313
  %317 = load double* %wt1, align 8
  %318 = load double* %rho, align 8
  %319 = fmul double %317, %318
  %320 = getelementptr inbounds [2 x double]* %u, i32 0, i64 0
  %321 = load double* %320, align 8
  %322 = load i32* %j10, align 4
  %323 = sext i32 %322 to i64
  %324 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %323
  %325 = load double* %324, align 8
  %326 = fmul double %321, %325
  %327 = getelementptr inbounds [2 x double]* %u, i32 0, i64 1
  %328 = load double* %327, align 8
  %329 = load i32* %j10, align 4
  %330 = add nsw i32 4, %329
  %331 = sext i32 %330 to i64
  %332 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %331
  %333 = load double* %332, align 8
  %334 = fmul double %328, %333
  %335 = fadd double %326, %334
  %336 = fmul double %319, %335
  %337 = load i32* %j10, align 4
  %338 = sext i32 %337 to i64
  %339 = load double*** %4, align 8
  %340 = getelementptr inbounds double** %339, i64 %338
  %341 = load double** %340, align 8
  %342 = getelementptr inbounds double* %341, i64 0
  %343 = load double* %342, align 8
  %344 = fadd double %343, %336
  store double %344, double* %342, align 8
  br label %345

; <label>:345                                     ; preds = %316
  %346 = load i32* %j10, align 4
  %347 = add nsw i32 %346, 1
  store i32 %347, i32* %j10, align 4
  br label %313

; <label>:348                                     ; preds = %313
  store i32 0, i32* %j11, align 4
  br label %349

; <label>:349                                     ; preds = %433, %348
  %350 = load i32* %j11, align 4
  %351 = icmp slt i32 %350, 4
  br i1 %351, label %352, label %436

; <label>:352                                     ; preds = %349
  store i32 0, i32* %k12, align 4
  br label %353

; <label>:353                                     ; preds = %429, %352
  %354 = load i32* %k12, align 4
  %355 = icmp slt i32 %354, 4
  br i1 %355, label %356, label %432

; <label>:356                                     ; preds = %353
  %357 = load double* %wt1, align 8
  %358 = load double* %rho, align 8
  %359 = fmul double %357, %358
  %360 = load i32* %j11, align 4
  %361 = sext i32 %360 to i64
  %362 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %361
  %363 = load double* %362, align 8
  %364 = load i32* %k12, align 4
  %365 = sext i32 %364 to i64
  %366 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %365
  %367 = load double* %366, align 8
  %368 = fmul double %363, %367
  %369 = load i32* %j11, align 4
  %370 = add nsw i32 4, %369
  %371 = sext i32 %370 to i64
  %372 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %371
  %373 = load double* %372, align 8
  %374 = load i32* %k12, align 4
  %375 = add nsw i32 4, %374
  %376 = sext i32 %375 to i64
  %377 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %376
  %378 = load double* %377, align 8
  %379 = fmul double %373, %378
  %380 = fadd double %368, %379
  %381 = fmul double %359, %380
  %382 = load double* %wt1, align 8
  %383 = load double* %rc2, align 8
  %384 = fmul double %382, %383
  %385 = getelementptr inbounds [2 x double]* %u, i32 0, i64 0
  %386 = load double* %385, align 8
  %387 = load i32* %j11, align 4
  %388 = sext i32 %387 to i64
  %389 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %388
  %390 = load double* %389, align 8
  %391 = fmul double %386, %390
  %392 = getelementptr inbounds [2 x double]* %u, i32 0, i64 1
  %393 = load double* %392, align 8
  %394 = load i32* %j11, align 4
  %395 = add nsw i32 4, %394
  %396 = sext i32 %395 to i64
  %397 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %396
  %398 = load double* %397, align 8
  %399 = fmul double %393, %398
  %400 = fadd double %391, %399
  %401 = fmul double %384, %400
  %402 = getelementptr inbounds [2 x double]* %u, i32 0, i64 0
  %403 = load double* %402, align 8
  %404 = load i32* %k12, align 4
  %405 = sext i32 %404 to i64
  %406 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %405
  %407 = load double* %406, align 8
  %408 = fmul double %403, %407
  %409 = getelementptr inbounds [2 x double]* %u, i32 0, i64 1
  %410 = load double* %409, align 8
  %411 = load i32* %k12, align 4
  %412 = add nsw i32 4, %411
  %413 = sext i32 %412 to i64
  %414 = getelementptr inbounds [8 x double]* %N_x, i32 0, i64 %413
  %415 = load double* %414, align 8
  %416 = fmul double %410, %415
  %417 = fadd double %408, %416
  %418 = fmul double %401, %417
  %419 = fsub double %381, %418
  %420 = load i32* %j11, align 4
  %421 = mul nsw i32 %420, 4
  %422 = load i32* %k12, align 4
  %423 = add nsw i32 %421, %422
  %424 = sext i32 %423 to i64
  %425 = load double** %3, align 8
  %426 = getelementptr inbounds double* %425, i64 %424
  %427 = load double* %426, align 8
  %428 = fadd double %427, %419
  store double %428, double* %426, align 8
  br label %429

; <label>:429                                     ; preds = %356
  %430 = load i32* %k12, align 4
  %431 = add nsw i32 %430, 1
  store i32 %431, i32* %k12, align 4
  br label %353

; <label>:432                                     ; preds = %353
  br label %433

; <label>:433                                     ; preds = %432
  %434 = load i32* %j11, align 4
  %435 = add nsw i32 %434, 1
  store i32 %435, i32* %j11, align 4
  br label %349

; <label>:436                                     ; preds = %349
  br label %437

; <label>:437                                     ; preds = %436
  %438 = load i32* %i, align 4
  %439 = add nsw i32 %438, 1
  store i32 %439, i32* %i, align 4
  br label %28

; <label>:440                                     ; preds = %28
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

declare double @pow(double, double) nounwind
"""

dirichlet_code = """
define void @dirichlet(double* %res) nounwind uwtable {
  %1 = alloca double*, align 8
  store double* %res, double** %1, align 8
  %2 = load double** %1, align 8
  store double 0.000000e+00, double* %2, align 8
  ret void
}
"""

init_cg_code = """
define void @init_cg(double* %r, double* %c, double* %u, double* %v, double* %p) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  %5 = alloca double*, align 8
  store double* %r, double** %1, align 8
  store double* %c, double** %2, align 8
  store double* %u, double** %3, align 8
  store double* %v, double** %4, align 8
  store double* %p, double** %5, align 8
  %6 = load double** %1, align 8
  %7 = load double* %6, align 8
  %8 = load double** %1, align 8
  %9 = load double* %8, align 8
  %10 = fmul double %7, %9
  %11 = load double** %2, align 8
  %12 = load double* %11, align 8
  %13 = fadd double %12, %10
  store double %13, double* %11, align 8
  %14 = load double** %1, align 8
  %15 = load double* %14, align 8
  %16 = load double** %5, align 8
  store double %15, double* %16, align 8
  %17 = load double** %3, align 8
  store double 0.000000e+00, double* %17, align 8
  %18 = load double** %4, align 8
  store double 0.000000e+00, double* %18, align 8
  ret void
}
"""

spMV_code = """
define void @spMV(double** %v, double* %K, double** %p) nounwind uwtable {
  %1 = alloca double**, align 8
  %2 = alloca double*, align 8
  %3 = alloca double**, align 8
  store double** %v, double*** %1, align 8
  store double* %K, double** %2, align 8
  store double** %p, double*** %3, align 8
  %4 = load double** %2, align 8
  %5 = getelementptr inbounds double* %4, i64 0
  %6 = load double* %5, align 8
  %7 = load double*** %3, align 8
  %8 = getelementptr inbounds double** %7, i64 0
  %9 = load double** %8, align 8
  %10 = getelementptr inbounds double* %9, i64 0
  %11 = load double* %10, align 8
  %12 = fmul double %6, %11
  %13 = load double*** %1, align 8
  %14 = getelementptr inbounds double** %13, i64 0
  %15 = load double** %14, align 8
  %16 = getelementptr inbounds double* %15, i64 0
  %17 = load double* %16, align 8
  %18 = fadd double %17, %12
  store double %18, double* %16, align 8
  %19 = load double** %2, align 8
  %20 = getelementptr inbounds double* %19, i64 1
  %21 = load double* %20, align 8
  %22 = load double*** %3, align 8
  %23 = getelementptr inbounds double** %22, i64 1
  %24 = load double** %23, align 8
  %25 = getelementptr inbounds double* %24, i64 0
  %26 = load double* %25, align 8
  %27 = fmul double %21, %26
  %28 = load double*** %1, align 8
  %29 = getelementptr inbounds double** %28, i64 0
  %30 = load double** %29, align 8
  %31 = getelementptr inbounds double* %30, i64 0
  %32 = load double* %31, align 8
  %33 = fadd double %32, %27
  store double %33, double* %31, align 8
  %34 = load double** %2, align 8
  %35 = getelementptr inbounds double* %34, i64 1
  %36 = load double* %35, align 8
  %37 = load double*** %3, align 8
  %38 = getelementptr inbounds double** %37, i64 0
  %39 = load double** %38, align 8
  %40 = getelementptr inbounds double* %39, i64 0
  %41 = load double* %40, align 8
  %42 = fmul double %36, %41
  %43 = load double*** %1, align 8
  %44 = getelementptr inbounds double** %43, i64 1
  %45 = load double** %44, align 8
  %46 = getelementptr inbounds double* %45, i64 0
  %47 = load double* %46, align 8
  %48 = fadd double %47, %42
  store double %48, double* %46, align 8
  %49 = load double** %2, align 8
  %50 = getelementptr inbounds double* %49, i64 2
  %51 = load double* %50, align 8
  %52 = load double*** %3, align 8
  %53 = getelementptr inbounds double** %52, i64 2
  %54 = load double** %53, align 8
  %55 = getelementptr inbounds double* %54, i64 0
  %56 = load double* %55, align 8
  %57 = fmul double %51, %56
  %58 = load double*** %1, align 8
  %59 = getelementptr inbounds double** %58, i64 0
  %60 = load double** %59, align 8
  %61 = getelementptr inbounds double* %60, i64 0
  %62 = load double* %61, align 8
  %63 = fadd double %62, %57
  store double %63, double* %61, align 8
  %64 = load double** %2, align 8
  %65 = getelementptr inbounds double* %64, i64 2
  %66 = load double* %65, align 8
  %67 = load double*** %3, align 8
  %68 = getelementptr inbounds double** %67, i64 0
  %69 = load double** %68, align 8
  %70 = getelementptr inbounds double* %69, i64 0
  %71 = load double* %70, align 8
  %72 = fmul double %66, %71
  %73 = load double*** %1, align 8
  %74 = getelementptr inbounds double** %73, i64 2
  %75 = load double** %74, align 8
  %76 = getelementptr inbounds double* %75, i64 0
  %77 = load double* %76, align 8
  %78 = fadd double %77, %72
  store double %78, double* %76, align 8
  %79 = load double** %2, align 8
  %80 = getelementptr inbounds double* %79, i64 3
  %81 = load double* %80, align 8
  %82 = load double*** %3, align 8
  %83 = getelementptr inbounds double** %82, i64 3
  %84 = load double** %83, align 8
  %85 = getelementptr inbounds double* %84, i64 0
  %86 = load double* %85, align 8
  %87 = fmul double %81, %86
  %88 = load double*** %1, align 8
  %89 = getelementptr inbounds double** %88, i64 0
  %90 = load double** %89, align 8
  %91 = getelementptr inbounds double* %90, i64 0
  %92 = load double* %91, align 8
  %93 = fadd double %92, %87
  store double %93, double* %91, align 8
  %94 = load double** %2, align 8
  %95 = getelementptr inbounds double* %94, i64 3
  %96 = load double* %95, align 8
  %97 = load double*** %3, align 8
  %98 = getelementptr inbounds double** %97, i64 0
  %99 = load double** %98, align 8
  %100 = getelementptr inbounds double* %99, i64 0
  %101 = load double* %100, align 8
  %102 = fmul double %96, %101
  %103 = load double*** %1, align 8
  %104 = getelementptr inbounds double** %103, i64 3
  %105 = load double** %104, align 8
  %106 = getelementptr inbounds double* %105, i64 0
  %107 = load double* %106, align 8
  %108 = fadd double %107, %102
  store double %108, double* %106, align 8
  %109 = load double** %2, align 8
  %110 = getelementptr inbounds double* %109, i64 5
  %111 = load double* %110, align 8
  %112 = load double*** %3, align 8
  %113 = getelementptr inbounds double** %112, i64 1
  %114 = load double** %113, align 8
  %115 = getelementptr inbounds double* %114, i64 0
  %116 = load double* %115, align 8
  %117 = fmul double %111, %116
  %118 = load double*** %1, align 8
  %119 = getelementptr inbounds double** %118, i64 1
  %120 = load double** %119, align 8
  %121 = getelementptr inbounds double* %120, i64 0
  %122 = load double* %121, align 8
  %123 = fadd double %122, %117
  store double %123, double* %121, align 8
  %124 = load double** %2, align 8
  %125 = getelementptr inbounds double* %124, i64 6
  %126 = load double* %125, align 8
  %127 = load double*** %3, align 8
  %128 = getelementptr inbounds double** %127, i64 2
  %129 = load double** %128, align 8
  %130 = getelementptr inbounds double* %129, i64 0
  %131 = load double* %130, align 8
  %132 = fmul double %126, %131
  %133 = load double*** %1, align 8
  %134 = getelementptr inbounds double** %133, i64 1
  %135 = load double** %134, align 8
  %136 = getelementptr inbounds double* %135, i64 0
  %137 = load double* %136, align 8
  %138 = fadd double %137, %132
  store double %138, double* %136, align 8
  %139 = load double** %2, align 8
  %140 = getelementptr inbounds double* %139, i64 6
  %141 = load double* %140, align 8
  %142 = load double*** %3, align 8
  %143 = getelementptr inbounds double** %142, i64 1
  %144 = load double** %143, align 8
  %145 = getelementptr inbounds double* %144, i64 0
  %146 = load double* %145, align 8
  %147 = fmul double %141, %146
  %148 = load double*** %1, align 8
  %149 = getelementptr inbounds double** %148, i64 2
  %150 = load double** %149, align 8
  %151 = getelementptr inbounds double* %150, i64 0
  %152 = load double* %151, align 8
  %153 = fadd double %152, %147
  store double %153, double* %151, align 8
  %154 = load double** %2, align 8
  %155 = getelementptr inbounds double* %154, i64 7
  %156 = load double* %155, align 8
  %157 = load double*** %3, align 8
  %158 = getelementptr inbounds double** %157, i64 3
  %159 = load double** %158, align 8
  %160 = getelementptr inbounds double* %159, i64 0
  %161 = load double* %160, align 8
  %162 = fmul double %156, %161
  %163 = load double*** %1, align 8
  %164 = getelementptr inbounds double** %163, i64 1
  %165 = load double** %164, align 8
  %166 = getelementptr inbounds double* %165, i64 0
  %167 = load double* %166, align 8
  %168 = fadd double %167, %162
  store double %168, double* %166, align 8
  %169 = load double** %2, align 8
  %170 = getelementptr inbounds double* %169, i64 7
  %171 = load double* %170, align 8
  %172 = load double*** %3, align 8
  %173 = getelementptr inbounds double** %172, i64 1
  %174 = load double** %173, align 8
  %175 = getelementptr inbounds double* %174, i64 0
  %176 = load double* %175, align 8
  %177 = fmul double %171, %176
  %178 = load double*** %1, align 8
  %179 = getelementptr inbounds double** %178, i64 3
  %180 = load double** %179, align 8
  %181 = getelementptr inbounds double* %180, i64 0
  %182 = load double* %181, align 8
  %183 = fadd double %182, %177
  store double %183, double* %181, align 8
  %184 = load double** %2, align 8
  %185 = getelementptr inbounds double* %184, i64 10
  %186 = load double* %185, align 8
  %187 = load double*** %3, align 8
  %188 = getelementptr inbounds double** %187, i64 2
  %189 = load double** %188, align 8
  %190 = getelementptr inbounds double* %189, i64 0
  %191 = load double* %190, align 8
  %192 = fmul double %186, %191
  %193 = load double*** %1, align 8
  %194 = getelementptr inbounds double** %193, i64 2
  %195 = load double** %194, align 8
  %196 = getelementptr inbounds double* %195, i64 0
  %197 = load double* %196, align 8
  %198 = fadd double %197, %192
  store double %198, double* %196, align 8
  %199 = load double** %2, align 8
  %200 = getelementptr inbounds double* %199, i64 11
  %201 = load double* %200, align 8
  %202 = load double*** %3, align 8
  %203 = getelementptr inbounds double** %202, i64 3
  %204 = load double** %203, align 8
  %205 = getelementptr inbounds double* %204, i64 0
  %206 = load double* %205, align 8
  %207 = fmul double %201, %206
  %208 = load double*** %1, align 8
  %209 = getelementptr inbounds double** %208, i64 2
  %210 = load double** %209, align 8
  %211 = getelementptr inbounds double* %210, i64 0
  %212 = load double* %211, align 8
  %213 = fadd double %212, %207
  store double %213, double* %211, align 8
  %214 = load double** %2, align 8
  %215 = getelementptr inbounds double* %214, i64 11
  %216 = load double* %215, align 8
  %217 = load double*** %3, align 8
  %218 = getelementptr inbounds double** %217, i64 2
  %219 = load double** %218, align 8
  %220 = getelementptr inbounds double* %219, i64 0
  %221 = load double* %220, align 8
  %222 = fmul double %216, %221
  %223 = load double*** %1, align 8
  %224 = getelementptr inbounds double** %223, i64 3
  %225 = load double** %224, align 8
  %226 = getelementptr inbounds double* %225, i64 0
  %227 = load double* %226, align 8
  %228 = fadd double %227, %222
  store double %228, double* %226, align 8
  %229 = load double** %2, align 8
  %230 = getelementptr inbounds double* %229, i64 15
  %231 = load double* %230, align 8
  %232 = load double*** %3, align 8
  %233 = getelementptr inbounds double** %232, i64 3
  %234 = load double** %233, align 8
  %235 = getelementptr inbounds double* %234, i64 0
  %236 = load double* %235, align 8
  %237 = fmul double %231, %236
  %238 = load double*** %1, align 8
  %239 = getelementptr inbounds double** %238, i64 3
  %240 = load double** %239, align 8
  %241 = getelementptr inbounds double* %240, i64 0
  %242 = load double* %241, align 8
  %243 = fadd double %242, %237
  store double %243, double* %241, align 8
  ret void
}
"""

dotPV_code = """
define void @dotPV(double* %p, double* %v, double* %c) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  store double* %p, double** %1, align 8
  store double* %v, double** %2, align 8
  store double* %c, double** %3, align 8
  %4 = load double** %1, align 8
  %5 = load double* %4, align 8
  %6 = load double** %2, align 8
  %7 = load double* %6, align 8
  %8 = fmul double %5, %7
  %9 = load double** %3, align 8
  %10 = load double* %9, align 8
  %11 = fadd double %10, %8
  store double %11, double* %9, align 8
  ret void
}
"""

updateUR_code = """
define void @updateUR(double* %u, double* %r, double* %p, double* %v, double* %alpha) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  %5 = alloca double*, align 8
  store double* %u, double** %1, align 8
  store double* %r, double** %2, align 8
  store double* %p, double** %3, align 8
  store double* %v, double** %4, align 8
  store double* %alpha, double** %5, align 8
  %6 = load double** %5, align 8
  %7 = load double* %6, align 8
  %8 = load double** %3, align 8
  %9 = load double* %8, align 8
  %10 = fmul double %7, %9
  %11 = load double** %1, align 8
  %12 = load double* %11, align 8
  %13 = fadd double %12, %10
  store double %13, double* %11, align 8
  %14 = load double** %5, align 8
  %15 = load double* %14, align 8
  %16 = load double** %4, align 8
  %17 = load double* %16, align 8
  %18 = fmul double %15, %17
  %19 = load double** %2, align 8
  %20 = load double* %19, align 8
  %21 = fsub double %20, %18
  store double %21, double* %19, align 8
  %22 = load double** %4, align 8
  store double 0.000000e+00, double* %22, align 8
  ret void
}
"""

dotR_code = """
define void @dotR(double* %r, double* %c) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  store double* %r, double** %1, align 8
  store double* %c, double** %2, align 8
  %3 = load double** %1, align 8
  %4 = load double* %3, align 8
  %5 = load double** %1, align 8
  %6 = load double* %5, align 8
  %7 = fmul double %4, %6
  %8 = load double** %2, align 8
  %9 = load double* %8, align 8
  %10 = fadd double %9, %7
  store double %10, double* %8, align 8
  ret void
}
"""

updateP_code = """
define void @updateP(double* %r, double* %p, double* %beta) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  store double* %r, double** %1, align 8
  store double* %p, double** %2, align 8
  store double* %beta, double** %3, align 8
  %4 = load double** %3, align 8
  %5 = load double* %4, align 8
  %6 = load double** %2, align 8
  %7 = load double* %6, align 8
  %8 = fmul double %5, %7
  %9 = load double** %1, align 8
  %10 = load double* %9, align 8
  %11 = fadd double %8, %10
  %12 = load double** %2, align 8
  store double %11, double* %12, align 8
  ret void
}
"""

update_code = """
define void @update(double* %phim, double* %res, double* %u, double* %rms) nounwind uwtable {
  %1 = alloca double*, align 8
  %2 = alloca double*, align 8
  %3 = alloca double*, align 8
  %4 = alloca double*, align 8
  store double* %phim, double** %1, align 8
  store double* %res, double** %2, align 8
  store double* %u, double** %3, align 8
  store double* %rms, double** %4, align 8
  %5 = load double** %3, align 8
  %6 = load double* %5, align 8
  %7 = load double** %1, align 8
  %8 = load double* %7, align 8
  %9 = fsub double %8, %6
  store double %9, double* %7, align 8
  %10 = load double** %2, align 8
  store double 0.000000e+00, double* %10, align 8
  %11 = load double** %3, align 8
  %12 = load double* %11, align 8
  %13 = load double** %3, align 8
  %14 = load double* %13, align 8
  %15 = fmul double %12, %14
  %16 = load double** %4, align 8
  %17 = load double* %16, align 8
  %18 = fadd double %17, %15
  store double %18, double* %16, align 8
  ret void
}
"""


llvm_kernel_opt = {'llvm_kernel': True}

dirichlet = Kernel(dirichlet_code, 'dirichlet', llvm_kernel_opt)

dotPV = Kernel(dotPV_code, 'dotPV', llvm_kernel_opt)

dotR = Kernel(dotR_code, 'dotR', llvm_kernel_opt)

init_cg = Kernel(init_cg_code, 'init_cg', llvm_kernel_opt)

res_calc = Kernel(res_calc_code, 'res_calc', llvm_kernel_opt)

spMV = Kernel(spMV_code, 'spMV', llvm_kernel_opt)

update = Kernel(update_code, 'update', llvm_kernel_opt)

updateP = Kernel(updateP_code, 'updateP', llvm_kernel_opt)

updateUR = Kernel(updateUR_code, 'updateUR', llvm_kernel_opt)
