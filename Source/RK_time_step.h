#ifndef __RK_TIME_STEP_H__
#define __RK_TIME_STEP_H__


#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "advection_2_3.h"

__global__ void intermediate_device(int N, real wight, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1, cudaComplex *RHS2, cudaComplex *RHS3, cudaComplex *RHS4, cudaComplex *x1_p, cudaComplex *x2_p, cudaComplex *x3_p, cudaComplex *x4_p);
__global__ void single_RK4_step_device(int N,  cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *RHS1_3, cudaComplex *RHS2_3, cudaComplex *RHS3_3, cudaComplex *RHS4_3, cudaComplex *RHS1_4, cudaComplex *RHS2_4, cudaComplex *RHS3_4, cudaComplex *RHS4_4, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat);


void RK4_single_step(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x1_p, cudaComplex *x2_p, cudaComplex *x3_p, cudaComplex *x4_p, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *RHS1_3, cudaComplex *RHS2_3, cudaComplex *RHS3_3, cudaComplex *RHS4_3, cudaComplex *RHS1_4, cudaComplex *RHS2_4, cudaComplex *RHS3_4, cudaComplex *RHS4_4);

void RightHandSide(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1, cudaComplex *RHS2, cudaComplex *RHS3, cudaComplex *RHS4, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d);

#endif
