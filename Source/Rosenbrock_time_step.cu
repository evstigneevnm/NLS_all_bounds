#include "Rosenbrock_time_step.h"

void print_matrix_extended(int batch_size, real *matrix, int s)
{
    for (int ii1 = 0;ii1 < matrix_size;++ii1) {
        for (int ii2 = 0;ii2 < extended_matrix_size;++ii2) {
            printf("%le,", matrix[IMB(s,ii1,ii2)]);
        }
        printf(";\n");
    }
}
void print_matrix(int batch_size, real *matrix, int s)
{
    for (int ii1 = 0;ii1 < matrix_size;++ii1) {
        for (int ii2 = 0;ii2 < matrix_size;++ii2) {
            printf("%le,", matrix[IMB(s,ii1,ii2)]);
        }
        printf(";\n");
    }
}

void print_solution(int batch_size, real *matrix, int s)
{
        printf("[");
        for (int ii1 = 0;ii1 < matrix_size;++ii1) {
                printf("%le,", matrix[IMB(s,ii1,extended_matrix_size-1)]); 
        }
        printf("]\n");

}


__global__ void ker_gauss_elim(int batch_size, real *m_in, real *m_out)
{   
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (!(i < batch_size)) return;

        real m[matrix_size][extended_matrix_size];          //matrix column

        #pragma unroll
        for (int ii1 = 0;ii1 < matrix_size;++ii1) {
                #pragma unroll
                for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) 
                    m[ii1][ii3] = m_in[IMB(i,ii1,ii3)];
        }

        //forward step
        #pragma unroll
        for (int ii1 = 0;ii1 < matrix_size;++ii1) {
                real diag = m[ii1][ii1];
                #pragma unroll
                for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) {
                        m[ii1][ii3] /= diag;
                }

                #pragma unroll
                for (int ii2 = 0;ii2 < matrix_size;++ii2) {
                        if (ii2 <= ii1) continue;
                        real mul = m[ii2][ii1];
                        #pragma unroll
                        for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) {
                                m[ii2][ii3] -= mul*m[ii1][ii3];
                        }
                }
        }

        //backward step
        #pragma unroll
        for (int ii1 = matrix_size-1;ii1 >= 0;--ii1) {
                #pragma unroll
                for (int ii2 = matrix_size-1;ii2 >= 0;--ii2) {
                        if (ii2 >= ii1) continue;
                        real mul = m[ii2][ii1];
                        #pragma unroll
                        for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) {
                                m[ii2][ii3] -= mul*m[ii1][ii3];
                        }
                }
        }

        #pragma unroll
        for (int ii1 = 0;ii1 < matrix_size;++ii1) {
                #pragma unroll
                for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) 
                    m_out[IMB(i,ii1,ii3)] = m[ii1][ii3];
        }
}


__global__ void set_batch_rhs_re(int batch_size, real *m_in, cudaComplex *x1, cudaComplex *x2, cudaComplex *x3, cudaComplex *x4){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (!(i < batch_size)) return;

        m_in[IMB(i,0,4)] = x1[i].x;
        m_in[IMB(i,1,4)] = x2[i].x;
        m_in[IMB(i,2,4)] = x3[i].x;
        m_in[IMB(i,3,4)] = x4[i].x;

}


__global__ void set_batch_rhs_imag(int batch_size, real *m_in, cudaComplex *x1, cudaComplex *x2, cudaComplex *x3, cudaComplex *x4){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (!(i < batch_size)) return;

        m_in[IMB(i,0,4)] = x1[i].y;
        m_in[IMB(i,1,4)] = x2[i].y;
        m_in[IMB(i,2,4)] = x3[i].y;
        m_in[IMB(i,3,4)] = x4[i].y;

}


__global__ void get_batch_sol_re(int batch_size, real *m_in, cudaComplex *x1, cudaComplex *x2, cudaComplex *x3, cudaComplex *x4){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (!(i < batch_size)) return;

        x1[i].x = m_in[IMB(i,0,4)];
        x2[i].x = m_in[IMB(i,1,4)];
        x3[i].x = m_in[IMB(i,2,4)];
        x4[i].x = m_in[IMB(i,3,4)];

}


__global__ void get_batch_sol_imag(int batch_size, real *m_in, cudaComplex *x1, cudaComplex *x2, cudaComplex *x3, cudaComplex *x4){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (!(i < batch_size)) return;

        x1[i].y = m_in[IMB(i,0,4)];
        x2[i].y = m_in[IMB(i,1,4)];
        x3[i].y = m_in[IMB(i,2,4)];
        x4[i].y = m_in[IMB(i,3,4)];

}

void construct_implicit_matrix(int batch_size, real *m_in, real g,  real *k_laplace, real dt, double a)
{
    
    for (int s = 0;s < batch_size;++s){
        //m_in[IMB(s,ii1,ii2)] = 
        real p = k_laplace[s];
        real q = -a*dt; //a_j*tau

        m_in[IMB(s,0,0)] = -lka;
        m_in[IMB(s,0,1)] = delta_betta - 0.5*p/betta;
        m_in[IMB(s,0,2)] = 0.0;
        m_in[IMB(s,0,3)] = -kappa;

        m_in[IMB(s,1,0)] = 0.5*p/betta - delta_betta;
        m_in[IMB(s,1,1)] = -lka;
        m_in[IMB(s,1,2)] = kappa;
        m_in[IMB(s,1,3)] = 0.0;

        m_in[IMB(s,2,0)] = 0.0;
        m_in[IMB(s,2,1)] = -kappa;
        m_in[IMB(s,2,2)] = g - lka;
        m_in[IMB(s,2,3)] = -delta_betta - 0.5*p/betta;

        m_in[IMB(s,3,0)] = kappa;
        m_in[IMB(s,3,1)] = 0.0;
        m_in[IMB(s,3,2)] = delta_betta + 0.5*p/betta;
        m_in[IMB(s,3,3)] = g - lka;
       
        //set RHS 
        m_in[IMB(s,0,4)] = 2.0;
        m_in[IMB(s,1,4)] = 3.0;
        m_in[IMB(s,2,4)] = 4.0;
        m_in[IMB(s,3,4)] = 5.0;

        for (int ii1 = 0;ii1 < matrix_size;++ii1){
            for (int ii2 = 0;ii2 < matrix_size;++ii2){
                m_in[IMB(s,ii1,ii2)]*=q; 
            }
        }
        for (int ii1 = 0;ii1 < matrix_size;++ii1){
            m_in[IMB(s,ii1,ii1)]+=1.0;  //diagonal
        }
    }
}


__global__ void get_rhs_step2_kernel(int M, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat, cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat)
{

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(j < M)) return;

        x1b_hat[j].x = x1_hat[j].x + ROSENBROCK_b21*RHS1_1[j].x;
        x1b_hat[j].y = x1_hat[j].y + ROSENBROCK_b21*RHS1_1[j].y;

        x2b_hat[j].x = x2_hat[j].x + ROSENBROCK_b21*RHS2_1[j].x;
        x2b_hat[j].y = x2_hat[j].y + ROSENBROCK_b21*RHS2_1[j].y;

        x3b_hat[j].x = x3_hat[j].x + ROSENBROCK_b21*RHS3_1[j].x;
        x3b_hat[j].y = x3_hat[j].y + ROSENBROCK_b21*RHS3_1[j].y;

        x4b_hat[j].x = x4_hat[j].x + ROSENBROCK_b21*RHS4_1[j].x;
        x4b_hat[j].y = x4_hat[j].y + ROSENBROCK_b21*RHS4_1[j].y;

        x1c_hat[j].x = x1_hat[j].x + ROSENBROCK_c21*RHS1_1[j].x;
        x1c_hat[j].y = x1_hat[j].y + ROSENBROCK_c21*RHS1_1[j].y;

        x2c_hat[j].x = x2_hat[j].x + ROSENBROCK_c21*RHS2_1[j].x;
        x2c_hat[j].y = x2_hat[j].y + ROSENBROCK_c21*RHS2_1[j].y;

        x3c_hat[j].x = x3_hat[j].x + ROSENBROCK_c21*RHS3_1[j].x;
        x3c_hat[j].y = x3_hat[j].y + ROSENBROCK_c21*RHS3_1[j].y;

        x4c_hat[j].x = x4_hat[j].x + ROSENBROCK_c21*RHS4_1[j].x;
        x4c_hat[j].y = x4_hat[j].y + ROSENBROCK_c21*RHS4_1[j].y;


}

__global__ void get_rhs_step3_kernel(int M, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat, cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat)
{

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(j < M)) return;

        x1b_hat[j].x = x1_hat[j].x + ROSENBROCK_b31*RHS1_1[j].x + ROSENBROCK_b32*RHS1_2[j].x;
        x1b_hat[j].y = x1_hat[j].y + ROSENBROCK_b31*RHS1_1[j].y + ROSENBROCK_b32*RHS1_2[j].y;

        x2b_hat[j].x = x2_hat[j].x + ROSENBROCK_b31*RHS2_1[j].x + ROSENBROCK_b32*RHS2_2[j].x;
        x2b_hat[j].y = x2_hat[j].y + ROSENBROCK_b31*RHS2_1[j].y + ROSENBROCK_b32*RHS2_2[j].y;

        x3b_hat[j].x = x3_hat[j].x + ROSENBROCK_b31*RHS3_1[j].x + ROSENBROCK_b32*RHS3_2[j].x;
        x3b_hat[j].y = x3_hat[j].y + ROSENBROCK_b31*RHS3_1[j].y + ROSENBROCK_b32*RHS3_2[j].y;

        x4b_hat[j].x = x4_hat[j].x + ROSENBROCK_b31*RHS4_1[j].x + ROSENBROCK_b32*RHS4_2[j].x;
        x4b_hat[j].y = x4_hat[j].y + ROSENBROCK_b31*RHS4_1[j].y + ROSENBROCK_b32*RHS4_2[j].y;

        x1c_hat[j].x = x1_hat[j].x + ROSENBROCK_c31*RHS1_1[j].x + ROSENBROCK_c32*RHS1_2[j].x;
        x1c_hat[j].y = x1_hat[j].y + ROSENBROCK_c31*RHS1_1[j].y + ROSENBROCK_c32*RHS1_2[j].y;

        x2c_hat[j].x = x2_hat[j].x + ROSENBROCK_c31*RHS2_1[j].x + ROSENBROCK_c32*RHS2_2[j].x;
        x2c_hat[j].y = x2_hat[j].y + ROSENBROCK_c31*RHS2_1[j].y + ROSENBROCK_c32*RHS2_2[j].y;

        x3c_hat[j].x = x3_hat[j].x + ROSENBROCK_c31*RHS3_1[j].x + ROSENBROCK_c32*RHS3_2[j].x;
        x3c_hat[j].y = x3_hat[j].y + ROSENBROCK_c31*RHS3_1[j].y + ROSENBROCK_c32*RHS3_2[j].y;

        x4c_hat[j].x = x4_hat[j].x + ROSENBROCK_c31*RHS4_1[j].x + ROSENBROCK_c32*RHS4_2[j].x;
        x4c_hat[j].y = x4_hat[j].y + ROSENBROCK_c31*RHS4_1[j].y + ROSENBROCK_c32*RHS4_2[j].y;


}


void get_rhs_vector_step2(dim3 dimGridI, dim3 dimBlockI, int M, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat, cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat)
{
    get_rhs_step2_kernel<<<dimGridI, dimBlockI>>>(M, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat);
}


void get_rhs_vector_step3(dim3 dimGridI, dim3 dimBlockI, int M, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1,cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat, cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat)
{
    get_rhs_step3_kernel<<<dimGridI, dimBlockI>>>(M, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat);

}


void init_matrices_Rosenbrock(int batch_size, real g,  real *k_laplace, real dt, real *&M1_, real *&M2_, real *&M3_, real *&M1_d_, real *&M2_d_, real *&M3_d_, real *&iM_d_, cudaComplex *&x1b_hat_, cudaComplex *&x2b_hat_, cudaComplex *&x3b_hat_, cudaComplex *&x4b_hat_, cudaComplex *&x1c_hat_, cudaComplex *&x2c_hat_, cudaComplex *&x3c_hat_, cudaComplex *&x4c_hat_)
{
    real *M1, *M2, *M3;
    real *M1_d, *M2_d, *M3_d, *iM_d;
    cudaComplex *x1b_hat, *x2b_hat, *x3b_hat, *x4b_hat, *x1c_hat, *x2c_hat, *x3c_hat, *x4c_hat;


    int all_size = matrix_size*extended_matrix_size*batch_size;
    allocate_real(all_size, 1, 1, 3, &M1, &M2, &M3);

    construct_implicit_matrix(batch_size, M1, g,  k_laplace, dt, ROSENBROCK_a1);
    construct_implicit_matrix(batch_size, M2, g,  k_laplace, dt, ROSENBROCK_a2);
    construct_implicit_matrix(batch_size, M3, g,  k_laplace, dt, ROSENBROCK_a3);    

    device_allocate_all_real(all_size, 1, 1, 4, &M1_d, &M2_d, &M3_d, &iM_d);
    device_from_host_all_real_cpy(all_size, {M1_d, M2_d, M3_d}, {M1, M2, M3});


    device_allocate_all_complex(batch_size, 1, 1, 8, &x1b_hat, &x2b_hat, &x3b_hat, &x4b_hat, &x1c_hat, &x2c_hat, &x3c_hat, &x4c_hat);

    M1_ = M1;
    M2_ = M2;
    M3_ = M3;
    M1_d_ = M1_d;
    M2_d_ = M2_d;
    M3_d_ = M3_d;
    iM_d_ = iM_d;
    
    x1b_hat_ = x1b_hat;
    x2b_hat_ = x2b_hat;
    x3b_hat_ = x3b_hat;
    x4b_hat_ = x4b_hat;
    x1c_hat_ = x1c_hat;
    x2c_hat_ = x2c_hat;
    x3c_hat_ = x3c_hat;
    x4c_hat_ = x4c_hat;

}

void clean_matrices_Rosenbrock(real *M1, real *M2, real *M3, real *M1_d, real *M2_d, real *M3_d, real *iM_d, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat, cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat)
{
    
    device_deallocate_all({M1_d, M2_d, M3_d, iM_d});
    device_deallocate_all({x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat});
    deallocate_real(3, M1, M2, M3);


}



void RightHandSide_distinct(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat, cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat, cudaComplex *RHS1, cudaComplex *RHS2, cudaComplex *RHS3, cudaComplex *RHS4, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d){

    calculate_convolution_2p3(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M,  x3c_hat, x4c_hat,x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d);

    assemble_NLS_device<<<dimGridI, dimBlockI>>>(M, dt, g, x1b_hat, x2b_hat, x3b_hat, x4b_hat, RHS1, RHS2, RHS3, RHS4, Q3_hat_mul, Q4_hat_mul, k_laplace_d);

}


__global__ void single_RB3_step_device(int M,  cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *RHS1_3, cudaComplex *RHS2_3, cudaComplex *RHS3_3, cudaComplex *RHS4_3, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(j < M)) return;

        x1_hat[j].x+=(ROSENBROCK_w1*RHS1_1[j].x+ROSENBROCK_w2*RHS1_2[j].x+ROSENBROCK_w3*RHS1_3[j].x);
        x1_hat[j].y+=(ROSENBROCK_w1*RHS1_1[j].y+ROSENBROCK_w2*RHS1_2[j].y+ROSENBROCK_w3*RHS1_3[j].y);

        x2_hat[j].x+=(ROSENBROCK_w1*RHS2_1[j].x+ROSENBROCK_w2*RHS2_2[j].x+ROSENBROCK_w3*RHS2_3[j].x);
        x2_hat[j].y+=(ROSENBROCK_w1*RHS2_1[j].y+ROSENBROCK_w2*RHS2_2[j].y+ROSENBROCK_w3*RHS2_3[j].y);     

        x3_hat[j].x+=(ROSENBROCK_w1*RHS3_1[j].x+ROSENBROCK_w2*RHS3_2[j].x+ROSENBROCK_w3*RHS3_3[j].x);
        x3_hat[j].y+=(ROSENBROCK_w1*RHS3_1[j].y+ROSENBROCK_w2*RHS3_2[j].y+ROSENBROCK_w3*RHS3_3[j].y);

        x4_hat[j].x+=(ROSENBROCK_w1*RHS4_1[j].x+ROSENBROCK_w2*RHS4_2[j].x+ROSENBROCK_w3*RHS4_3[j].x);
        x4_hat[j].y+=(ROSENBROCK_w1*RHS4_1[j].y+ROSENBROCK_w2*RHS4_2[j].y+ROSENBROCK_w3*RHS4_3[j].y);


}


void RB3_single_step(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x1_p, cudaComplex *x2_p, cudaComplex *x3_p, cudaComplex *x4_p, cudaComplex *x1b_hat, cudaComplex *x2b_hat, cudaComplex *x3b_hat, cudaComplex *x4b_hat,cudaComplex *x1c_hat, cudaComplex *x2c_hat, cudaComplex *x3c_hat, cudaComplex *x4c_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *RHS1_3, cudaComplex *RHS2_3, cudaComplex *RHS3_3, cudaComplex *RHS4_3, real *M1_d, real *M2_d, real *M3_d, real *iM_d)
{

    RightHandSide_distinct(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_hat, x1_hat, x2_hat, x3_hat, x4_hat, x1_p, x2_p, x3_p, x4_p, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

    
    set_batch_rhs_re<<<dimGridI, dimBlockI>>>(M, M1_d, x1_p, x2_p, x3_p, x4_p);
    ker_gauss_elim<<<dimGridI, dimBlockI>>>(M, M1_d, iM_d);
    get_batch_sol_re<<<dimGridI, dimBlockI>>>(M, iM_d, RHS1_1, RHS2_1, RHS3_1, RHS4_1);
    set_batch_rhs_imag<<<dimGridI, dimBlockI>>>(M, M1_d, x1_p, x2_p, x3_p, x4_p);
    ker_gauss_elim<<<dimGridI, dimBlockI>>>(M, M1_d, iM_d);
    get_batch_sol_imag<<<dimGridI, dimBlockI>>>(M, iM_d, RHS1_1, RHS2_1, RHS3_1, RHS4_1);

    get_rhs_vector_step2(dimGridI, dimBlockI, M, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat);

    RightHandSide_distinct(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat, x1_p, x2_p, x3_p, x4_p, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

    set_batch_rhs_re<<<dimGridI, dimBlockI>>>(M, M2_d, x1_p, x2_p, x3_p, x4_p);
    ker_gauss_elim<<<dimGridI, dimBlockI>>>(M, M2_d, iM_d);
    get_batch_sol_re<<<dimGridI, dimBlockI>>>(M, iM_d, RHS1_2, RHS2_2, RHS3_2, RHS4_2);
    set_batch_rhs_imag<<<dimGridI, dimBlockI>>>(M, M2_d, x1_p, x2_p, x3_p, x4_p);
    ker_gauss_elim<<<dimGridI, dimBlockI>>>(M, M2_d, iM_d);
    get_batch_sol_imag<<<dimGridI, dimBlockI>>>(M, iM_d, RHS1_2, RHS2_2, RHS3_2, RHS4_2);

    get_rhs_vector_step3(dimGridI, dimBlockI, M, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat);   
    
    RightHandSide_distinct(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat, x1_p, x2_p, x3_p, x4_p, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

    set_batch_rhs_re<<<dimGridI, dimBlockI>>>(M, M3_d, x1_p, x2_p, x3_p, x4_p);
    ker_gauss_elim<<<dimGridI, dimBlockI>>>(M, M3_d, iM_d);
    get_batch_sol_re<<<dimGridI, dimBlockI>>>(M, iM_d, RHS1_3, RHS2_3, RHS3_3, RHS4_3);
    set_batch_rhs_imag<<<dimGridI, dimBlockI>>>(M, M3_d, x1_p, x2_p, x3_p, x4_p);
    ker_gauss_elim<<<dimGridI, dimBlockI>>>(M, M3_d, iM_d);
    get_batch_sol_imag<<<dimGridI, dimBlockI>>>(M, iM_d, RHS1_3, RHS2_3, RHS3_3, RHS4_3);


    single_RB3_step_device<<<dimGridI, dimBlockI>>>(M,  RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, RHS1_3, RHS2_3, RHS3_3, RHS4_3, x1_hat, x2_hat, x3_hat, x4_hat);

}

void check_gauss(dim3 dimGridI, dim3 dimBlockI, int batch_size, real g,  real *k_laplace, real dt, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat)
{

    int s = 0;
    real *M1, *M2, *M3, *m_out;
    real *M1_d, *M2_d, *M3_d, *m_out_d;
    real *iM1_d, *iM2_d, *iM3_d;
    real *iM1, *iM2, *iM3;
    int all_size = matrix_size*extended_matrix_size*batch_size;

    allocate_real(all_size, 1, 1, 4, &M1, &M2, &M3, &m_out);
    allocate_real(all_size, 1, 1, 3, &iM1, &iM2, &iM3);

    construct_implicit_matrix(batch_size, M1, g,  k_laplace, dt, ROSENBROCK_a1);
    construct_implicit_matrix(batch_size, M2, g,  k_laplace, dt, ROSENBROCK_a2);
    construct_implicit_matrix(batch_size, M3, g,  k_laplace, dt, ROSENBROCK_a3);

    device_allocate_all_real(all_size, 1, 1, 4, &M1_d, &M2_d, &M3_d, &m_out_d);
    device_allocate_all_real(all_size, 1, 1, 3, &iM1_d, &iM2_d, &iM3_d);
    //device_allocate_all_real(all_size, {M1_d, M2_d, M3_d, m_out_d});
    device_from_host_all_real_cpy(all_size, {M1_d, M2_d, M3_d}, {M1, M2, M3});
    
    //set_batch_rhs_re<<<dimGridI, dimBlockI>>>(batch_size, M1_d, x1_hat, x2_hat, x3_hat, x4_hat);

    host_from_device_all_real_cpy(all_size, {M1}, {M1_d});
    host_from_device_all_real_cpy(all_size, {M2}, {M1_d});
    host_from_device_all_real_cpy(all_size, {M3}, {M1_d});
    //print matrices:
    print_matrix(batch_size, M1, s);
    printf("\n");
    print_matrix(batch_size, M2, s);
    printf("\n");
    print_matrix(batch_size, M3, s);
    printf("\n");

    ker_gauss_elim<<<dimGridI, dimBlockI>>>(batch_size, M1_d, iM1_d);
    ker_gauss_elim<<<dimGridI, dimBlockI>>>(batch_size, M2_d, iM2_d);
    ker_gauss_elim<<<dimGridI, dimBlockI>>>(batch_size, M3_d, iM3_d);

    //get_batch_sol_re<<<dimGridI, dimBlockI>>>(batch_size, iM1_d, x1_hat, x2_hat, x3_hat, x4_hat);

    host_from_device_all_real_cpy(all_size, {iM1, iM2, iM3}, {iM1_d, iM2_d, iM3_d});
    device_deallocate_all({M1_d, M2_d, M3_d, m_out_d, iM1_d, iM2_d, iM3_d});

    //print matrices:
    //print_matrix_extended(batch_size, iM1, s);
    
    print_solution(batch_size, iM1, s);
    print_solution(batch_size, iM2, s);
    print_solution(batch_size, iM3, s);


    deallocate_real(7, M1, M2, M3, m_out, iM1, iM2, iM3);

}