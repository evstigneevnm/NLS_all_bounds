#ifndef __JACOBIAN_H__
#define __JACOBIAN_H__


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Macros.h"
#include "RK_time_step.h"
#include "file_operations.h"
#include "cuda_supp.h"
#include "memory_operations.h"
#include "math_support.h"


void print_Jacobian(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_plus, cudaComplex *RHS1_minus, cudaComplex *Diff_RHS1, cudaComplex *RHS2_plus, cudaComplex *RHS2_minus, cudaComplex *Diff_RHS2, cudaComplex *RHS3_plus, cudaComplex *RHS3_minus, cudaComplex *Diff_RHS3, cudaComplex *RHS4_plus, cudaComplex *RHS4_minus, cudaComplex *Diff_RHS4);

void build_Jacobian(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_plus, cudaComplex *RHS1_minus, cudaComplex *Diff_RHS1, cudaComplex *RHS2_plus, cudaComplex *RHS2_minus, cudaComplex *Diff_RHS2, cudaComplex *RHS3_plus, cudaComplex *RHS3_minus, cudaComplex *Diff_RHS3, cudaComplex *RHS4_plus, cudaComplex *RHS4_minus, cudaComplex *Diff_RHS4, real* Jacobian_d, cudaComplex *x1_eps, cudaComplex *x2_eps, cudaComplex *x3_eps, cudaComplex *x4_eps);

#endif