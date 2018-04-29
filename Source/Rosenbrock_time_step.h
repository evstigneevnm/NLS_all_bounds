#ifndef __Rosenbrock_time_step_H__
#define __Rosenbrock_time_step_H__

#include "Macros.h"
#include "cuda_supp.h"
#include "advection_2_3.h"
#include "memory_operations.h"
//for the RightHandSide
#include "RK_time_step.h"

#define ROSENBROCK_a1 0.7970967740096232
#define ROSENBROCK_a2 0.5913813968007854
#define ROSENBROCK_a3 0.1347052663841181
#define ROSENBROCK_c21 1.058925354610082
#define ROSENBROCK_c31 0.5
#define ROSENBROCK_c32 -0.3759391872875334
#define ROSENBROCK_b21 8.0/7.0
#define ROSENBROCK_b31 71.0/254.0
#define ROSENBROCK_b32 7.0/36.0
#define ROSENBROCK_w1 0.125
#define ROSENBROCK_w2 0.125
#define ROSENBROCK_w3 0.75

//matrix sizes
#define matrix_size 4
#define extended_matrix_size (matrix_size + 1)

// batch indexing
#ifndef IMB
    #define IMB(m,j,k)  ((j)*(batch_size)*(extended_matrix_size)+(k)*(batch_size)+(m))
#endif


void init_matrices_Rosenbrock(int batch_size, real g,  real *k_laplace, real dt, real *&M1_, real *&M2_, real *&M3_, real *&M1_d_, real *&M2_d_, real *&M3_d_, real *&iM_d_, cudaComplex *&x1b_hat_, cudaComplex *&x2b_hat_, cudaComplex *&x3b_hat_, cudaComplex *&x4b_hat_, cudaComplex *&x1c_hat_, cudaComplex *&x2c_hat_, cudaComplex *&x3c_hat_, cudaComplex *&x4c_hat_);

void clean_matrices_Rosenbrock(real *M1, real *M2, real *M3, real *M1_d, real *M2_d, real *M3_d, real *iM_d, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat, cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat);

void RB3_single_step(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x1_p, cudaComplex *x2_p, cudaComplex *x3_p, cudaComplex *x4_p, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat,cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *RHS1_3, cudaComplex *RHS2_3, cudaComplex *RHS3_3, cudaComplex *RHS4_3, real *M1_d, real *M2_d, real *M3_d, real *iM_d);



#endif