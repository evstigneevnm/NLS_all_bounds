#include "Jacobian.h"




__global__ void DX_j_device(cudaComplex *xN_hat_source, cudaComplex *xN_hat_destination, real eps_re, real eps_im, int N, int coordinate){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int flag=0;
	
	if(j<N){
		
		if((coordinate-j)==0)
			flag=1;
		else
			flag=0;

		xN_hat_destination[j].x=xN_hat_source[j].x+eps_re*flag;
		xN_hat_destination[j].y=xN_hat_source[j].y+eps_im*flag;

		
	}



}


__global__ void Diff_RHS_device(cudaComplex *RHS1_plus, cudaComplex *RHS1_minus, cudaComplex *Diff_RHS1, cudaComplex *RHS2_plus, cudaComplex *RHS2_minus, cudaComplex *Diff_RHS2, cudaComplex *RHS3_plus, cudaComplex *RHS3_minus, cudaComplex *Diff_RHS3, cudaComplex *RHS4_plus, cudaComplex *RHS4_minus, cudaComplex *Diff_RHS4, real eps, int N){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	
	if(j<N){
		
		Diff_RHS1[j].x=(RHS1_plus[j].x-RHS1_minus[j].x)/(2*eps);
		Diff_RHS1[j].y=(RHS1_plus[j].y-RHS1_minus[j].y)/(2*eps);
		Diff_RHS2[j].x=(RHS2_plus[j].x-RHS2_minus[j].x)/(2*eps);
		Diff_RHS2[j].y=(RHS2_plus[j].y-RHS2_minus[j].y)/(2*eps);	
		Diff_RHS3[j].x=(RHS3_plus[j].x-RHS3_minus[j].x)/(2*eps);
		Diff_RHS3[j].y=(RHS3_plus[j].y-RHS3_minus[j].y)/(2*eps);
		Diff_RHS4[j].x=(RHS4_plus[j].x-RHS4_minus[j].x)/(2*eps);
		Diff_RHS4[j].y=(RHS4_plus[j].y-RHS4_minus[j].y)/(2*eps);

	}

}

__global__ void Jacobian_Colomn_device(cudaComplex *Diff_RHS1,  cudaComplex *Diff_RHS2, cudaComplex *Diff_RHS3, cudaComplex *Diff_RHS4, real *Jacobian_d, int N, int Jacobian_Row){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	int Ny=N*2*4;
	if(j<N){
		
		real factor=1;
		//if((j>N/4-1)&&(j<3*N/4)) factor=0;

		Jacobian_d[I(Jacobian_Row,j)]=Diff_RHS1[j].x*factor;
		Jacobian_d[I(Jacobian_Row,j+N)]=Diff_RHS1[j].y*factor;
		Jacobian_d[I(Jacobian_Row,j+2*N)]=Diff_RHS2[j].x*factor;
		Jacobian_d[I(Jacobian_Row,j+3*N)]=Diff_RHS2[j].y*factor;
		Jacobian_d[I(Jacobian_Row,j+4*N)]=Diff_RHS3[j].x*factor;
		Jacobian_d[I(Jacobian_Row,j+5*N)]=Diff_RHS3[j].y*factor;
		Jacobian_d[I(Jacobian_Row,j+6*N)]=Diff_RHS4[j].x*factor;
		Jacobian_d[I(Jacobian_Row,j+7*N)]=Diff_RHS4[j].y*factor;
	}

}



__global__ void Jacobian_per_Raw_device(cudaComplex *Diff_RHS1,  cudaComplex *Diff_RHS2, cudaComplex *Diff_RHS3, cudaComplex *Diff_RHS4, real *Jacobian_d, int N, int Jacobian_Row){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	int Ny=N*2*4;
	if(j<N){
		

		real factor=1.0, add_value=0.0;
		//if((j>N/4)&&(j<3*N/4)){ factor=0000.0; add_value=0000.0; }
		//if((Jacobian_Row>8*(N/4))&&(Jacobian_Row<8*(3*N/4))){ factor=0000.0; add_value=000.0; }

		int j_Jacobian=8*j+0;
		Jacobian_d[I(Jacobian_Row,j_Jacobian)]=Diff_RHS1[j].x*factor+add_value;
		j_Jacobian=8*j+1;
		Jacobian_d[I(Jacobian_Row,j_Jacobian)]=Diff_RHS1[j].y*factor+add_value;
		j_Jacobian=8*j+2;
		Jacobian_d[I(Jacobian_Row,j_Jacobian)]=Diff_RHS2[j].x*factor+add_value;
		j_Jacobian=8*j+3;
		Jacobian_d[I(Jacobian_Row,j_Jacobian)]=Diff_RHS2[j].y*factor+add_value;
		j_Jacobian=8*j+4;
		Jacobian_d[I(Jacobian_Row,j_Jacobian)]=Diff_RHS3[j].x*factor+add_value;
		j_Jacobian=8*j+5;
		Jacobian_d[I(Jacobian_Row,j_Jacobian)]=Diff_RHS3[j].y*factor+add_value;
		j_Jacobian=8*j+6;
		Jacobian_d[I(Jacobian_Row,j_Jacobian)]=Diff_RHS4[j].x*factor+add_value;
		j_Jacobian=8*j+7;
		Jacobian_d[I(Jacobian_Row,j_Jacobian)]=Diff_RHS4[j].y*factor+add_value;
	
	}

}



void build_Jacobian(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_plus, cudaComplex *RHS1_minus, cudaComplex *Diff_RHS1, cudaComplex *RHS2_plus, cudaComplex *RHS2_minus, cudaComplex *Diff_RHS2, cudaComplex *RHS3_plus, cudaComplex *RHS3_minus, cudaComplex *Diff_RHS3, cudaComplex *RHS4_plus, cudaComplex *RHS4_minus, cudaComplex *Diff_RHS4, real* Jacobian_d, cudaComplex *x1_eps, cudaComplex *x2_eps, cudaComplex *x3_eps, cudaComplex *x4_eps){

	real eps=5.0e-5;
	int JN=(2*4*M);
	dt=1.0;
	for(int t=0;t<M;t++){

		//x1, Re
		DX_j_device<<<dimGridI, dimBlockI>>>(x1_hat, x1_eps, eps, 0, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_eps, x2_hat, x3_hat, x4_hat, RHS1_plus, RHS2_plus, RHS3_plus, RHS4_plus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		DX_j_device<<<dimGridI, dimBlockI>>>(x1_hat, x1_eps, -eps, 0, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_eps, x2_hat, x3_hat, x4_hat, RHS1_minus, RHS2_minus, RHS3_minus, RHS4_minus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		Diff_RHS_device<<<dimGridI, dimBlockI>>>(RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, eps, M);
		//Jacobian_Colomn_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, N, t*8+0);
		Jacobian_per_Raw_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, M, t*8+0);
		//x1, Im
		DX_j_device<<<dimGridI, dimBlockI>>>(x1_hat, x1_eps, 0, eps, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_eps, x2_hat, x3_hat, x4_hat, RHS1_plus, RHS2_plus, RHS3_plus, RHS4_plus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);
		DX_j_device<<<dimGridI, dimBlockI>>>(x1_hat, x1_eps, 0, -eps, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_eps, x2_hat, x3_hat, x4_hat, RHS1_minus, RHS2_minus, RHS3_minus, RHS4_minus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		Diff_RHS_device<<<dimGridI, dimBlockI>>>(RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, eps, M);
		//Jacobian_Colomn_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, N, t*8+1);
		Jacobian_per_Raw_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, M, t*8+1);
		//x2, Re
		DX_j_device<<<dimGridI, dimBlockI>>>(x2_hat, x2_eps, eps, 0, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_eps, x3_hat, x4_hat, RHS1_plus, RHS2_plus, RHS3_plus, RHS4_plus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);
		DX_j_device<<<dimGridI, dimBlockI>>>(x2_hat, x2_eps, -eps, 0, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_eps, x3_hat, x4_hat, RHS1_minus, RHS2_minus, RHS3_minus, RHS4_minus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		Diff_RHS_device<<<dimGridI, dimBlockI>>>(RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, eps, M);
		//Jacobian_Colomn_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, N, t*8+2);
		Jacobian_per_Raw_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, M, t*8+2);

		//x2, Im
		DX_j_device<<<dimGridI, dimBlockI>>>(x2_hat, x2_eps, 0, eps, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_eps, x3_hat, x4_hat, RHS1_plus, RHS2_plus, RHS3_plus, RHS4_plus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);
		DX_j_device<<<dimGridI, dimBlockI>>>(x2_hat, x2_eps, 0, -eps, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_eps, x3_hat, x4_hat, RHS1_minus, RHS2_minus, RHS3_minus, RHS4_minus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		Diff_RHS_device<<<dimGridI, dimBlockI>>>(RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, eps, M);
		//Jacobian_Colomn_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, N, t*8+3);
		Jacobian_per_Raw_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, M, t*8+3);

		//x3, Re
		DX_j_device<<<dimGridI, dimBlockI>>>(x3_hat, x3_eps, eps, 0, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_eps, x4_hat, RHS1_plus, RHS2_plus, RHS3_plus, RHS4_plus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);
		DX_j_device<<<dimGridI, dimBlockI>>>(x3_hat, x3_eps, -eps, 0, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_eps, x4_hat, RHS1_minus, RHS2_minus, RHS3_minus, RHS4_minus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		Diff_RHS_device<<<dimGridI, dimBlockI>>>(RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, eps, M);
		//Jacobian_Colomn_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, N, t*8+4);
		Jacobian_per_Raw_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, M, t*8+4);

		//x3, Im
		DX_j_device<<<dimGridI, dimBlockI>>>(x3_hat, x3_eps, 0, eps, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_eps, x4_hat, RHS1_plus, RHS2_plus, RHS3_plus, RHS4_plus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);
		DX_j_device<<<dimGridI, dimBlockI>>>(x3_hat, x3_eps, 0, -eps, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_eps, x4_hat, RHS1_minus, RHS2_minus, RHS3_minus, RHS4_minus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		Diff_RHS_device<<<dimGridI, dimBlockI>>>(RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, eps, M);
		//Jacobian_Colomn_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, N, t*8+5);
		Jacobian_per_Raw_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, M, t*8+5);

		//x4, Re
		DX_j_device<<<dimGridI, dimBlockI>>>(x4_hat, x4_eps, eps, 0, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_eps, RHS1_plus, RHS2_plus, RHS3_plus, RHS4_plus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);
		DX_j_device<<<dimGridI, dimBlockI>>>(x4_hat, x4_eps, -eps, 0, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_eps, RHS1_minus, RHS2_minus, RHS3_minus, RHS4_minus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		Diff_RHS_device<<<dimGridI, dimBlockI>>>(RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, eps, M);
		//Jacobian_Colomn_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, N, t*8+6);
		Jacobian_per_Raw_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, M, t*8+6);

		//x4, Im
		DX_j_device<<<dimGridI, dimBlockI>>>(x4_hat, x4_eps, 0, eps, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_eps, RHS1_plus, RHS2_plus, RHS3_plus, RHS4_plus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);
		DX_j_device<<<dimGridI, dimBlockI>>>(x4_hat, x4_eps, 0, -eps, M, t);
		RightHandSide(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_eps, RHS1_minus, RHS2_minus, RHS3_minus, RHS4_minus, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d);

		Diff_RHS_device<<<dimGridI, dimBlockI>>>(RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, eps, M);
		//Jacobian_Colomn_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, N, t*8+7);
		Jacobian_per_Raw_device<<<dimGridI, dimBlockI>>>(Diff_RHS1,  Diff_RHS2, Diff_RHS3, Diff_RHS4, Jacobian_d, M, t*8+7);

		//printf("[%.03f\%]  \r",100*(t+1)/(1.0*N));	

	}



	
}


void print_Jacobian(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M, real dt, real g, cudaComplex *x1_hat, cudaComplex *x2_hat, cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d, real *k_laplace_d, cudaComplex *RHS1_plus, cudaComplex *RHS1_minus, cudaComplex *Diff_RHS1, cudaComplex *RHS2_plus, cudaComplex *RHS2_minus, cudaComplex *Diff_RHS2, cudaComplex *RHS3_plus, cudaComplex *RHS3_minus, cudaComplex *Diff_RHS3, cudaComplex *RHS4_plus, cudaComplex *RHS4_minus, cudaComplex *Diff_RHS4){

	real *Jacobian;
	real *Jacobian_d;
	int JN=(2*4*M);
	cudaComplex *x1_eps, *x2_eps, *x3_eps, *x4_eps;
	allocate_real(JN, JN, 1, 1, &Jacobian);
	device_allocate_all_real(JN, JN, 1, 1, &Jacobian_d);
	device_allocate_all_complex(M, 1, 1, 4,&x1_eps,&x2_eps,&x3_eps,&x4_eps);

	build_Jacobian(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_hat, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_plus, RHS1_minus, Diff_RHS1, RHS2_plus, RHS2_minus, Diff_RHS2, RHS3_plus, RHS3_minus, Diff_RHS3, RHS4_plus, RHS4_minus, Diff_RHS4, Jacobian_d, x1_eps, x2_eps, x3_eps, x4_eps);


	host_device_real_cpy(Jacobian, Jacobian_d, JN, JN, 1);

	device_deallocate_all_complex(4, x1_eps,x2_eps,x3_eps,x4_eps);
	device_deallocate_all_real(1, Jacobian_d);

	write_file_matrix("Jacobian.dat", Jacobian, JN, JN);


	printf("\n");
	deallocate_real(1, Jacobian);
	
}


 
