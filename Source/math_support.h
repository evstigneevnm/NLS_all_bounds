#ifndef __MATH_SUPPORT_H__
#define __MATH_SUPPORT_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include "Macros.h"

void FFTN_Device(cufftHandle planR2C, real *source, cudaComplex *destination);
void iFFTN_Device(cufftHandle planC2R, cudaComplex *source, real *destination);
void two_real_2_one_complex(dim3 dimBlock, dim3 dimGrid, real* f_Re, real* f_Im, cudaComplex *fc, int N);
void one_complex_2_two_real(dim3 dimBlock, dim3 dimGrid, cudaComplex *fc, real* f_Re, real* f_Im, int N);
void all_real_2_complex(dim3 dimBlock, dim3 dimGrid, real* x1_Re_d, real* x1_Im_d, cudaComplex *x1_hat, real* x2_Re_d, real* x2_Im_d, cudaComplex *x2_hat, real* x3_Re_d, real* x3_Im_d, cudaComplex *x3_hat, real* x4_Re_d, real* x4_Im_d, cudaComplex *x4_hat, int N);
void all_complex_2_real(dim3 dimBlock, dim3 dimGrid, real* x1_Re_d, real* x1_Im_d, cudaComplex *x1_hat, real* x2_Re_d, real* x2_Im_d, cudaComplex *x2_hat, real* x3_Re_d, real* x3_Im_d, cudaComplex *x3_hat, real* x4_Re_d, real* x4_Im_d, cudaComplex *x4_hat, int N);

void Domain_To_Image(cufftHandle planR2C, real* x1_d, cudaComplex *x1_hat, real* x2_d, cudaComplex *x2_hat, real* x3_d, cudaComplex *x3_hat, real* x4_d, cudaComplex *x4_hat);
void Image_to_Domain(dim3 dimGridD, dim3 dimBlockD, cufftHandle planC2R, int N, real* x1_d, cudaComplex *x1_hat, real* x2_d, cudaComplex *x2_hat, real* x3_d, cudaComplex *x3_hat, real* x4_d, cudaComplex *x4_hat);
void build_Laplace_Wavenumbers(int N,  real L, real *k_laplace, char boundary);
void build_mask_matrix(int N,  real L, real *mask_2_3, char boundary);


#endif
