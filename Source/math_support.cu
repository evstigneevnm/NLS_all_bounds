#include "math_support.h"


//cuda math common functions

/*
__global__ void double2complex_device(real* f, cudaComplex *fc, int N){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j<N){
		fc[j].x=f[j];
		fc[j].y=0.0;
	}
}

__global__ void complex2double_device(cudaComplex *fc, real* f, int N){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j<N){
		
		f[j]=fc[j].x/(1.0*N);
		
	}
}

*/


__global__ void scale_double_device(real* f, int N){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j<N){
		
		f[j]=f[j]/(1.0*N);
		
	}
}


__global__ void two_real_2_one_complex_device(real* f_Re, real* f_Im, cudaComplex *fc, int N){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j<N){
		fc[j].x=f_Re[j];
		fc[j].y=f_Im[j];
		
	}
}

__global__ void one_complex_2_two_real_device(cudaComplex *fc, real* f_Re, real* f_Im, int N){
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j<N){
		f_Re[j]=fc[j].x;
		f_Im[j]=fc[j].y;
		
	}
}

/*
void double2complex(dim3 dimBlock, dim3 dimGrid, real* f, cudaComplex *fc, int N){

	double2complex_device<<<dimGrid, dimBlock>>>(f, fc, N);
}

void complex2double(dim3 dimBlock, dim3 dimGrid, cudaComplex *fc, real* f, int N){

	complex2double_device<<<dimGrid, dimBlock>>>(fc, f, N);
}

*/

void two_real_2_one_complex(dim3 dimBlock, dim3 dimGrid, real* f_Re, real* f_Im, cudaComplex *fc, int N){

	two_real_2_one_complex_device<<<dimGrid, dimBlock>>>(f_Re, f_Im, fc, N);
}

void one_complex_2_two_real(dim3 dimBlock, dim3 dimGrid, cudaComplex *fc, real* f_Re, real* f_Im, int N){

	one_complex_2_two_real_device<<<dimGrid, dimBlock>>>(fc, f_Re, f_Im, N);
}


void all_real_2_complex(dim3 dimBlockI, dim3 dimGridI, real* x1_Re_d, real* x1_Im_d, cudaComplex *x1_hat, real* x2_Re_d, real* x2_Im_d, cudaComplex *x2_hat, real* x3_Re_d, real* x3_Im_d, cudaComplex *x3_hat, real* x4_Re_d, real* x4_Im_d, cudaComplex *x4_hat, int N){

	two_real_2_one_complex(dimBlockI, dimGridI, x1_Re_d, x1_Im_d, x1_hat, N);
	two_real_2_one_complex(dimBlockI, dimGridI, x2_Re_d, x2_Im_d, x2_hat, N);
	two_real_2_one_complex(dimBlockI, dimGridI, x3_Re_d, x3_Im_d, x3_hat, N);
	two_real_2_one_complex(dimBlockI, dimGridI, x4_Re_d, x4_Im_d, x4_hat, N);	

}

void all_complex_2_real(dim3 dimBlockI, dim3 dimGridI, real* x1_Re_d, real* x1_Im_d, cudaComplex *x1_hat, real* x2_Re_d, real* x2_Im_d, cudaComplex *x2_hat, real* x3_Re_d, real* x3_Im_d, cudaComplex *x3_hat, real* x4_Re_d, real* x4_Im_d, cudaComplex *x4_hat, int M){

	one_complex_2_two_real(dimBlockI, dimGridI, x1_hat, x1_Re_d, x1_Im_d, M);
	one_complex_2_two_real(dimBlockI, dimGridI, x2_hat, x2_Re_d, x2_Im_d, M);
	one_complex_2_two_real(dimBlockI, dimGridI, x3_hat, x3_Re_d, x3_Im_d, M);
	one_complex_2_two_real(dimBlockI, dimGridI, x4_hat, x4_Re_d, x4_Im_d, M);	

}




void FFTN_Device(cufftHandle planR2C, real *source, cudaComplex *destination){
cufftResult result;

/*
CUFFT_SUCCESS 	cuFFT successfully executed the FFT plan.
CUFFT_INVALID_PLAN 	The plan parameter is not a valid handle.
CUFFT_INVALID_VALUE 	At least one of the parameters idata and odata is not valid.
CUFFT_INTERNAL_ERROR 	An internal driver error was detected.
CUFFT_EXEC_FAILED 	cuFFT failed to execute the transform on the GPU.
CUFFT_SETUP_FAILED 	The cuFFT library failed to initialize.

Read more at: http://docs.nvidia.com/cuda/cufft/index.html#ixzz3eNyIE1pY
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
*/

	result=cufftExecRtoC(planR2C, source, destination);
	if (result != CUFFT_SUCCESS) { 
		fprintf (stderr,"CUFFT Forward failed:\n"); 
		if(result==CUFFT_INVALID_PLAN) fprintf (stderr,"The plan parameter is not a valid handle.\n"); 
		else if(result==CUFFT_INVALID_VALUE) fprintf (stderr,"At least one of the parameters idata and odata is not valid.\n"); 
		else if(result==CUFFT_INTERNAL_ERROR) fprintf (stderr,"An internal driver error was detected.\n"); 
		else if(result==CUFFT_EXEC_FAILED) fprintf (stderr,"cuFFT failed to execute the transform on the GPU.\n"); 		
		else if(result==CUFFT_SETUP_FAILED) fprintf (stderr,"The cuFFT library failed to initialize.\n");
		else fprintf (stderr,"Unknown error!\n");

		exit(EXIT_FAILURE);
	}
}


void iFFTN_Device(cufftHandle planC2R, cudaComplex *source, real *destination){
cufftResult result;
	result=cufftExecCtoR(planC2R, source, destination);
	if (result != CUFFT_SUCCESS) { 
		fprintf (stderr,"CUFFT Inverse failed\n"); 
		if(result==CUFFT_INVALID_PLAN) fprintf (stderr,"The plan parameter is not a valid handle.\n"); 
		else if(result==CUFFT_INVALID_VALUE) fprintf (stderr,"At least one of the parameters idata and odata is not valid.\n"); 
		else if(result==CUFFT_INTERNAL_ERROR) fprintf (stderr,"An internal driver error was detected.\n"); 
		else if(result==CUFFT_EXEC_FAILED) fprintf (stderr,"cuFFT failed to execute the transform on the GPU.\n"); 		
		else if(result==CUFFT_SETUP_FAILED) fprintf (stderr,"The cuFFT library failed to initialize.\n");
		else fprintf (stderr,"Unknown error!\n");

		exit(EXIT_FAILURE); 
	}
}




void Domain_To_Image(cufftHandle planR2C, real* x1_d, cudaComplex *x1_hat, real* x2_d, cudaComplex *x2_hat, real* x3_d, cudaComplex *x3_hat, real* x4_d, cudaComplex *x4_hat){


	FFTN_Device(planR2C, x1_d, x1_hat);
	FFTN_Device(planR2C, x2_d, x2_hat);
	FFTN_Device(planR2C, x3_d, x3_hat);
	FFTN_Device(planR2C, x4_d, x4_hat);			

}



void Image_to_Domain(dim3 dimGridD, dim3 dimBlockD, cufftHandle planC2R, int N, real* x1_d, cudaComplex *x1_hat, real* x2_d, cudaComplex *x2_hat, real* x3_d, cudaComplex *x3_hat, real* x4_d, cudaComplex *x4_hat ){

	iFFTN_Device(planC2R,x1_hat, x1_d);
	iFFTN_Device(planC2R,x2_hat, x2_d);
	iFFTN_Device(planC2R,x3_hat, x3_d);
	iFFTN_Device(planC2R,x4_hat, x4_d);

	scale_double_device<<<dimGridD, dimBlockD>>>(x1_d, N);
	scale_double_device<<<dimGridD, dimBlockD>>>(x2_d, N);
	scale_double_device<<<dimGridD, dimBlockD>>>(x3_d, N);
	scale_double_device<<<dimGridD, dimBlockD>>>(x4_d, N);
}



//host math functions
void build_Laplace_Wavenumbers(int N,  real L, real *k_laplace){

	for(int j=0;j<N; j++){
		int m=j;
		if(j>=N/2)
			m=j-N;
		k_laplace[j]=-2.0*PI/L*m*2.0*PI/L*m;

	}


}




void build_mask_matrix(int N,  real L, real *mask_2_3){

int m;
	real kxMax=N*PI/L;
	for(int j=0;j<N;j++){
		m=j;
		if(j>=N/2)
			m=j-N;
			
		real kx=m*PI/L;



		real sphere2=kx*kx/kxMax/kxMax;
		//	2/3 limitation!
		if(sphere2<4.0/9.0){
			mask_2_3[j]=1.0;
		}
		else{
			mask_2_3[j]=0.0; //!
		}
	
	}	


}



