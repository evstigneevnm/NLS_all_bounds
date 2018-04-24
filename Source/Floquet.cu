#include "Floquet.h"

/*
This function performs the matrix-matrix multiplication:
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)

Read more at: http://docs.nvidia.com/cuda/cublas/index.html#ixzz3dpazz8jr
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook


This function performs the matrix-matrix addition/transposition:
cublasStatus_t cublasDgeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const double          *alpha,
                          const double          *A, int lda,
                          const double          *beta,
                          const double          *B, int ldb,
                          double          *C, int ldc)


Read more at: http://docs.nvidia.com/cuda/cublas/index.html#ixzz3dpbuiaWt
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook



*/
void MatrixMatrixMul(cublasHandle_t handle, cublasStatus_t status, int N, real *A, real *B, real *C, real alpha, real beta){


	status=cublasMM(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
	if (status!= CUBLAS_STATUS_SUCCESS){
			printf("cublasMM returned error code %d, line(%d)\n", status, __LINE__);
			exit(EXIT_FAILURE);
	}


}

void MatrixMatrixSum(cublasHandle_t handle, cublasStatus_t status, int N, real *A, real *B, real *C, real wight){
	const double alpha=1.0;

	status=cublasMpM(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, &alpha, A, N, &wight, B, N, C, N);
	if (status != CUBLAS_STATUS_SUCCESS){
			printf("cublasMpM returned error code %d, line(%d)\n", status, __LINE__);
			exit(EXIT_FAILURE);
	}

}



void MatrixMatrixSum_full(cublasHandle_t handle, cublasStatus_t status, int N, real *A, real *B, real *C, real alpha, real beta){


	status=cublasMpM(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, &alpha, A, N, &beta, B, N, C, N);
	if (status != CUBLAS_STATUS_SUCCESS){
			printf("cublasMpM returned error code %d, line(%d)\n", status, __LINE__);
			exit(EXIT_FAILURE);
	}

}


__global__ void make_ident_device(int N, int k, real *Ident){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	int Ny=N*2*4;
	if(j<N){
		for(int dj=0;dj<8;dj++){
			for(int dk=0;dk<8;dk++){
				int index_k=8*k+dk;
				int index_j=8*j+dj;
				Ident[I(index_k,index_j)]=0;
				if(index_k==index_j)
					Ident[I(index_k,index_j)]=1;
				
				}
		}

	}

}

__global__ void make_zero_device(int N, int k, real *Ident){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	int Ny=N*2*4;
	if(j<N){
		for(int dj=0;dj<8;dj++){
			for(int dk=0;dk<8;dk++){
				int index_k=8*k+dk;
				int index_j=8*j+dj;
				Ident[I(index_k,index_j)]=0;
				
				}
		}

	}

}


__global__ void copy_matrix_device(int N, int k, real *source, real *destination){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	int Ny=N*2*4;
	if(j<N){
		for(int dj=0;dj<8;dj++){
			for(int dk=0;dk<8;dk++){
				int index_k=8*k+dk;
				int index_j=8*j+dj;
				destination[I(index_k,index_j)]=source[I(index_k,index_j)];
				
				}
		}

	}

}


void calculate_K_Step(cublasHandle_t handle, cublasStatus_t status, int N, real *Jacobian_d, real *Xj_d, real *Kj_d, real dt){

	int Ny=N*2*4;
	MatrixMatrixMul(handle, status, Ny, Jacobian_d, Xj_d, Kj_d, dt, 0);

}

void calculate_X_Step(cublasHandle_t handle, cublasStatus_t status, int N, real *X0_d, real *Kj_d, real *Xp_d, real wight){

	int Ny=N*2*4;
	MatrixMatrixSum(handle, status, Ny, X0_d, Kj_d, Xp_d, wight);

}


void calculate_RK4_Step(dim3 dimBlock, dim3 dimGrid, cublasHandle_t handle, cublasStatus_t status, int N, real *X0_d, real *X1_d, real *Xp_d, real *K1_d, real *K2_d, real *K3_d, real *K4_d){
	
	int Ny=N*2*4;

	MatrixMatrixSum_full(handle, status, Ny, K1_d, K4_d, Xp_d, 1.0/6.0, 1.0/6.0);
	MatrixMatrixSum_full(handle, status, Ny, Xp_d, K2_d, X1_d, 1.0, 1.0/3.0);
	MatrixMatrixSum_full(handle, status, Ny, X1_d, K3_d, Xp_d, 1.0, 1.0/3.0);
	MatrixMatrixSum_full(handle, status, Ny, Xp_d, X0_d, X1_d, 1.0, 1.0);

	for(int k=0;k<N;k++){
		copy_matrix_device<<<dimGrid, dimBlock>>>(N, k, X1_d, X0_d);
	}


}


void print_Floquet(int T, dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x1_p, cudaComplex *x2_p, cudaComplex *x3_p, cudaComplex *x4_p, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_1, cudaComplex *RHS2_1, cudaComplex *RHS3_1, cudaComplex *RHS4_1, cudaComplex *RHS1_2, cudaComplex *RHS2_2, cudaComplex *RHS3_2, cudaComplex *RHS4_2, cudaComplex *RHS1_3, cudaComplex *RHS2_3, cudaComplex *RHS3_3, cudaComplex *RHS4_3, cudaComplex *RHS1_4, cudaComplex *RHS2_4, cudaComplex *RHS3_4, cudaComplex *RHS4_4){

	int JN=(2*4*M);
	//host
	real *Monodromy;
	//device
	real *Jacobian_d;
	real *X0_d, *X1_d, *E_d, *Xp_d;
	real *K1_d, *K2_d, *K3_d, *K4_d;
	cudaComplex *x1_eps, *x2_eps, *x3_eps, *x4_eps;
	cudaComplex *RHS1_plus, *RHS1_minus, *Diff_RHS1,  *RHS2_plus,  *RHS2_minus, *Diff_RHS2,  *RHS3_plus,  *RHS3_minus, *Diff_RHS3, *RHS4_plus, *RHS4_minus, *Diff_RHS4;


	cublasHandle_t handle;
	cublasStatus_t status;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("cublasCreate returned error code %d, line(%d)\n", status, __LINE__);
		exit(EXIT_FAILURE);
	}

	allocate_real(JN, JN, 1, 1, &Monodromy);
	device_allocate_all_real(JN, JN, 1, 9, &Jacobian_d, &X0_d, &X1_d, &E_d, &K1_d, &K2_d, &K3_d, &K4_d, &Xp_d);
	device_allocate_all_complex(M, 1, 1, 4,&x1_eps,&x2_eps,&x3_eps,&x4_eps);
	device_allocate_all_complex(M, 1, 1, 12, &RHS1_plus, &RHS1_minus, &Diff_RHS1, &RHS2_plus,  &RHS2_minus, &Diff_RHS2,  &RHS3_plus,  &RHS3_minus, &Diff_RHS3, &RHS4_plus, &RHS4_minus, &Diff_RHS4);

	for(int k=0;k<M;k++){
		make_ident_device<<<dimGridI, dimBlockI>>>(M, k, X1_d);
		make_ident_device<<<dimGridI, dimBlockI>>>(M, k, X0_d);
	}

	printf("\n");

	real check_val=0.0;

	for(int t=0;t<T;t++){


//K1:
		build_Jacobian(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, 1.0, g, x1_hat, x2_hat, x3_hat, x4_hat, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, Jacobian_d, x1_eps, x2_eps, x3_eps, x4_eps);

		calculate_K_Step(handle, status, M, Jacobian_d, X0_d, K1_d, dt);
		calculate_X_Step(handle, status, M, X0_d, K1_d, Xp_d, 0.5);

		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		intermediate_device<<<dimGridI, dimBlockI>>>(M, 0.5, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, x1_p, x2_p, x3_p, x4_p);


//K2:	
		build_Jacobian(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, 1.0, g, x1_p, x2_p, x3_p, x4_p, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, Jacobian_d, x1_eps, x2_eps, x3_eps, x4_eps);

		calculate_K_Step(handle, status, M, Jacobian_d, Xp_d, K2_d, dt);
		calculate_X_Step(handle, status, M, X0_d, K2_d, Xp_d, 0.5);	



		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_p, x2_p, x3_p, x4_p, RHS1_2, RHS2_2, RHS3_2, RHS4_2, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		intermediate_device<<<dimGridI, dimBlockI>>>(M, 0.5, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_2, RHS2_2, RHS3_2, RHS4_2, x1_p, x2_p, x3_p, x4_p);

//K3:	
		build_Jacobian(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, 1.0, g, x1_p, x2_p, x3_p, x4_p, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, Jacobian_d, x1_eps, x2_eps, x3_eps, x4_eps);

		calculate_K_Step(handle, status, M, Jacobian_d, Xp_d, K3_d, dt);
		calculate_X_Step(handle, status, M, X0_d, K3_d, Xp_d, 1.0);	


		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_p, x2_p, x3_p, x4_p, RHS1_3, RHS2_3, RHS3_3, RHS4_3, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		intermediate_device<<<dimGridI, dimBlockI>>>(M, 1.0, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_3, RHS2_3, RHS3_3, RHS4_3, x1_p, x2_p, x3_p, x4_p);

//K4:	
		build_Jacobian(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, 1.0, g, x1_p, x2_p, x3_p, x4_p, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, Jacobian_d, x1_eps, x2_eps, x3_eps, x4_eps);

		calculate_K_Step(handle, status, M, Jacobian_d, Xp_d, K4_d, dt);

		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_p, x2_p, x3_p, x4_p, RHS1_4, RHS2_4, RHS3_4, RHS4_4, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);


//RK-4 assembly:
		single_RK4_step_device<<<dimGridI, dimBlockI>>>(M,  RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, RHS1_3, RHS2_3, RHS3_3, RHS4_3, RHS1_4, RHS2_4, RHS3_4, RHS4_4, x1_hat, x2_hat, x3_hat, x4_hat);
		calculate_RK4_Step(dimBlockI, dimGridI, handle, status, M, X0_d, X1_d, Xp_d, K1_d, K2_d, K3_d, K4_d);


		host_device_real_cpy(&check_val, &(X1_d[0]),1,1,1);
		printf("Monodromy Matrix construction [%.03f\%] Check: %.06le \r",100*(t+1)/(1.0*T),check_val);	


	}


	status=cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("cublasDestroy returned error code %d, line(%d)\n", status, __LINE__);
		exit(EXIT_FAILURE);
	}



	host_device_real_cpy(Monodromy, X1_d, JN, JN, 1);
	device_deallocate_all_complex(4, x1_eps,x2_eps,x3_eps,x4_eps);
	device_deallocate_all_complex(12, RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus,  RHS2_minus, Diff_RHS2,  RHS3_plus,  RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4);
	device_deallocate_all_real(9, Jacobian_d, X0_d, X1_d, E_d, K1_d, K2_d, K3_d, K4_d, Xp_d);

	write_file_matrix("Monodromy.dat", Monodromy, JN, JN);

	deallocate_real(1, Monodromy);
	printf("\n");

}
