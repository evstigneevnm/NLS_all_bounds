#include "RK_time_step.h"




__global__ void copy_arrays_device(int N, cudaComplex *source1, cudaComplex *source2, cudaComplex *source3, cudaComplex *source4, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3, cudaComplex *destination4){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){

        destination1[j].x=source1[j].x;
        destination1[j].y=source1[j].y;

        destination2[j].x=source2[j].x;
        destination2[j].y=source2[j].y;

        destination3[j].x=source3[j].x;
        destination3[j].y=source3[j].y;


    }

}



__global__ void single_RK4_step_device(int N,  cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *RHS1_3, cudaComplex *RHS2_3, cudaComplex *RHS3_3, cudaComplex *RHS4_3, cudaComplex *RHS1_4, cudaComplex *RHS2_4, cudaComplex *RHS3_4, cudaComplex *RHS4_4, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){

        x1_hat[j].x+=(RHS1_1[j].x+2.0*RHS1_2[j].x+2.0*RHS1_3[j].x+RHS1_4[j].x)/6.0;
        x1_hat[j].y+=(RHS1_1[j].y+2.0*RHS1_2[j].y+2.0*RHS1_3[j].y+RHS1_4[j].y)/6.0;

        x2_hat[j].x+=(RHS2_1[j].x+2.0*RHS2_2[j].x+2.0*RHS2_3[j].x+RHS2_4[j].x)/6.0;
        x2_hat[j].y+=(RHS2_1[j].y+2.0*RHS2_2[j].y+2.0*RHS2_3[j].y+RHS2_4[j].y)/6.0;     

        x3_hat[j].x+=(RHS3_1[j].x+2.0*RHS3_2[j].x+2.0*RHS3_3[j].x+RHS3_4[j].x)/6.0;
        x3_hat[j].y+=(RHS3_1[j].y+2.0*RHS3_2[j].y+2.0*RHS3_3[j].y+RHS3_4[j].y)/6.0;

        x4_hat[j].x+=(RHS4_1[j].x+2.0*RHS4_2[j].x+2.0*RHS4_3[j].x+RHS4_4[j].x)/6.0;
        x4_hat[j].y+=(RHS4_1[j].y+2.0*RHS4_2[j].y+2.0*RHS4_3[j].y+RHS4_4[j].y)/6.0;

    }

}



__global__ void assemble_NLS_device(int N, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1, cudaComplex *RHS2, cudaComplex *RHS3, cudaComplex *RHS4, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *k_laplace_d){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){

        RHS1[j].x=dt*(-0.5*k_laplace_d[j]/betta*x2_hat[j].x-(lka*x1_hat[j].x-delta_betta*x2_hat[j].x+kappa*x4_hat[j].x));
        RHS1[j].y=dt*(-0.5*k_laplace_d[j]/betta*x2_hat[j].y-(lka*x1_hat[j].y-delta_betta*x2_hat[j].y+kappa*x4_hat[j].y));

        RHS2[j].x=dt*(0.5*k_laplace_d[j]/betta*x1_hat[j].x-(delta_betta*x1_hat[j].x+lka*x2_hat[j].x-kappa*x3_hat[j].x));
        RHS2[j].y=dt*(0.5*k_laplace_d[j]/betta*x1_hat[j].y-(delta_betta*x1_hat[j].y+lka*x2_hat[j].y-kappa*x3_hat[j].y));

        RHS3[j].x=dt*(-0.5*k_laplace_d[j]/betta*x4_hat[j].x-(kappa*x2_hat[j].x+(lka-g)*x3_hat[j].x+delta_betta*x4_hat[j].x)-Q3_hat_mul[j].x);   
        RHS3[j].y=dt*(-0.5*k_laplace_d[j]/betta*x4_hat[j].y-(kappa*x2_hat[j].y+(lka-g)*x3_hat[j].y+delta_betta*x4_hat[j].y)-Q3_hat_mul[j].y);

        RHS4[j].x=dt*(0.5*k_laplace_d[j]/betta*x3_hat[j].x-(-kappa*x1_hat[j].x+(lka-g)*x4_hat[j].x-delta_betta*x3_hat[j].x)-Q4_hat_mul[j].x);   
        RHS4[j].y=dt*(0.5*k_laplace_d[j]/betta*x3_hat[j].y-(-kappa*x1_hat[j].y+(lka-g)*x4_hat[j].y-delta_betta*x3_hat[j].y)-Q4_hat_mul[j].y);   
    }

}



__global__ void intermediate_device(int N, real wight, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1, cudaComplex *RHS2, cudaComplex *RHS3, cudaComplex *RHS4, cudaComplex *x1_p, cudaComplex *x2_p, cudaComplex *x3_p, cudaComplex *x4_p){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){

        x1_p[j].x=x1_hat[j].x+wight*RHS1[j].x;
        x2_p[j].x=x2_hat[j].x+wight*RHS2[j].x;
        x3_p[j].x=x3_hat[j].x+wight*RHS3[j].x;
        x4_p[j].x=x4_hat[j].x+wight*RHS4[j].x;


        x1_p[j].y=x1_hat[j].y+wight*RHS1[j].y;
        x2_p[j].y=x2_hat[j].y+wight*RHS2[j].y;
        x3_p[j].y=x3_hat[j].y+wight*RHS3[j].y;
        x4_p[j].y=x4_hat[j].y+wight*RHS4[j].y;

    }

}





void RightHandSide(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1, cudaComplex *RHS2, cudaComplex *RHS3, cudaComplex *RHS4, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d){

    calculate_convolution_2p3(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M,  x3_hat, x4_hat,x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d);

    assemble_NLS_device<<<dimGridI, dimBlockI>>>(M, dt, g, x1_hat, x2_hat, x3_hat, x4_hat, RHS1, RHS2, RHS3, RHS4, Q3_hat_mul, Q4_hat_mul, k_laplace_d);

}



void RK4_single_step(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x1_p, cudaComplex *x2_p, cudaComplex *x3_p, cudaComplex *x4_p, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *RHS1_3, cudaComplex *RHS2_3, cudaComplex *RHS3_3, cudaComplex *RHS4_3, cudaComplex *RHS1_4, cudaComplex *RHS2_4, cudaComplex *RHS3_4, cudaComplex *RHS4_4){


//K1:
    RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

    intermediate_device<<<dimGridI, dimBlockI>>>(M, 0.5, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, x1_p, x2_p, x3_p, x4_p);
//K2:   
    RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_p, x2_p, x3_p, x4_p, RHS1_2, RHS2_2, RHS3_2, RHS4_2, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

    intermediate_device<<<dimGridI, dimBlockI>>>(M, 0.5, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_2, RHS2_2, RHS3_2, RHS4_2, x1_p, x2_p, x3_p, x4_p);
//K3:   
    RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_p, x2_p, x3_p, x4_p, RHS1_3, RHS2_3, RHS3_3, RHS4_3, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

    intermediate_device<<<dimGridI, dimBlockI>>>(M, 1.0, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_3, RHS2_3, RHS3_3, RHS4_3, x1_p, x2_p, x3_p, x4_p);

//K4:   
    RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_p, x2_p, x3_p, x4_p, RHS1_4, RHS2_4, RHS3_4, RHS4_4, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);


//RK-4 assembly:
    single_RK4_step_device<<<dimGridI, dimBlockI>>>(M,  RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, RHS1_3, RHS2_3, RHS3_3, RHS4_3, RHS1_4, RHS2_4, RHS3_4, RHS4_4, x1_hat, x2_hat, x3_hat, x4_hat);

}