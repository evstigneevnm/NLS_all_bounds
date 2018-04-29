#ifndef __MACROS_H__
#define __MACROS_H__

#include "constants.h"

#define BLOCKSIZE 64
//#define block_size_x 16
//#define block_size_y BLOCKSIZE/block_size_x


#define FS for ( j=0 ; j<N ; j++ ){
#define FE }


#ifndef I
	#define I(i,j) (i)*(Ny)+(j)
#endif



//* Type redifinition!
#define real double

#define cudaComplex cufftDoubleComplex
#define cudaReal cufftDoubleReal

#define ComplexToComplex CUFFT_Z2Z
#define RealToComplex CUFFT_D2Z
#define ComplexToReal CUFFT_Z2D

#define cufftExecXtoX cufftExecZ2Z
#define cufftExecRtoC cufftExecD2Z
#define cufftExecCtoR cufftExecZ2D

#define cublasMM cublasDgemm
#define cublasMpM cublasDgeam

//*/
/*
#define real float
#define cudaComplex cufftComplex
#define ComplexToComplex CUFFT_C2C
#define cufftExecXtoX  cufftExecC2C
//*/



#define PI 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128

#define min2(X,Y) ((X) < (Y) ? (X) : (Y)) 
#define max2(X,Y) ((X) < (Y) ? (Y) : (X)) 
#define max3(X,Y,Z) max2(max2(X,Y),max2(Y,Z))

#define Labs(X) ((X) < 0 ? (X):-(X))

#endif
