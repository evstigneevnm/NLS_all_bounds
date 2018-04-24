#include "advection_2_3.h"






__global__ void copy_complex_device(int N, cudaComplex *source1, cudaComplex *source2, cudaComplex *destination1, cudaComplex *destination2){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j<N){

	
		destination1[j].x=source1[j].x;
		destination1[j].y=source1[j].y;
		
		destination2[j].x=source2[j].x;
		destination2[j].y=source2[j].y;

	}

}




__global__ void scale_range_device(int N, real *x3_c, real *x4_c){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j<N){

		x3_c[j]=x3_c[j]/(1.0*N);
		x4_c[j]=x4_c[j]/(1.0*N);
		

	}

}



__global__ void calculate_Qs_device(int N, real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul){

unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j<N){

		real modulus2=x3_c[j]*x3_c[j]+x4_c[j]*x4_c[j];
		
		Q3_mul[j]=ImfGamma*modulus2*x3_c[j]+RefGamma*modulus2*x4_c[j];
		Q4_mul[j]=-RefGamma*modulus2*x3_c[j]+ImfGamma*modulus2*x4_c[j];



	}	
}




__global__ void filter_1p3_wavenumbers_device(int N, cudaComplex *x3_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat, cudaComplex *x4_hat_cut, real *mask_2_3_d){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j<N){
		
		x3_hat_cut[j].x=x3_hat[j].x*mask_2_3_d[j];
		x3_hat_cut[j].y=x3_hat[j].y*mask_2_3_d[j];
		
		x4_hat_cut[j].x=x4_hat[j].x*mask_2_3_d[j];
		x4_hat_cut[j].y=x4_hat[j].y*mask_2_3_d[j];	
		
		
	}


}





void filter_wavenumbers(dim3 dimGrid, dim3 dimBlock, int N, cudaComplex *x3_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat, cudaComplex *x4_hat_cut, real *mask_2_3_d){
	
	// reduced hat

	filter_1p3_wavenumbers_device<<<dimGrid, dimBlock>>>(N, x3_hat, x3_hat_cut, x4_hat, x4_hat_cut, mask_2_3_d);


}



void convolute_2p3(dim3 dimGridD, dim3 dimBlockD, cufftHandle planR2C, cufftHandle planC2R, int N, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul){
	//go to Domain
	iFFTN_Device(planC2R, x3_hat_cut, x3_c);	
	iFFTN_Device(planC2R, x4_hat_cut, x4_c);

	scale_range_device<<<dimGridD, dimBlockD>>>(N, x3_c, x4_c);
	
	calculate_Qs_device<<<dimGridD, dimBlockD>>>(N, x3_c, x4_c, Q3_mul, Q4_mul);

	//return to Image
	FFTN_Device(planR2C, Q3_mul, Q3_hat_mul);
	FFTN_Device(planR2C, Q4_mul, Q4_hat_mul);



}



void calculate_convolution_2p3(dim3 dimGridD, dim3 dimBlockD, dim3 dimGridI, dim3 dimBlockI, cufftHandle planR2C, cufftHandle planC2R, int N, int M,  cudaComplex *x3_hat, cudaComplex *x4_hat, cudaComplex *x3_hat_cut, cudaComplex *x4_hat_cut,  real *x3_c, real *x4_c, real *Q3_mul, real *Q4_mul, cudaComplex *Q3_hat_mul, cudaComplex *Q4_hat_mul, real *mask_2_3_d){

	//reduce velocity and derivatives wavenumbers 
	filter_wavenumbers(dimGridI, dimBlockI, M, x3_hat, x3_hat_cut, x4_hat, x4_hat_cut, mask_2_3_d);

	convolute_2p3(dimGridD, dimBlockD, planR2C, planC2R, N, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul);


}

