#ifndef __H_ADVECTION_H__
#define __H_ADVECTION_H__

#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "math_support.h"



void calculate_convolution_2p3(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M,  cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d);


#endif
