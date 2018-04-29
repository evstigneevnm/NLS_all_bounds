#include "memory_operations.h"


void allocate_d(int size, real **array){
    *array=(real*)malloc(sizeof(real)*size);
    if ( !array ){
        fprintf(stderr,"\n unable to allocate real memory!\n");
        exit(-1);
    }
    // else{
    //     for(int j=0;j<size;j++)
    //         array[j]=0.0;
    // }
}



real* allocate_d(int Nx, int Ny, int Nz){
	int size=(Nx)*(Ny)*(Nz);
	real* array;
	array=(real*)malloc(sizeof(real)*size);
	if ( !array ){
		fprintf(stderr,"\n unable to allocate real memory!\n");
		exit(-1);
	}
	else{
		for(int j=0;j<size;j++)
				array[j]=0.0;
	}
	
	return array;
}

int* allocate_i(int Nx, int Ny, int Nz){
	int size=(Nx)*(Ny)*(Nz);
	int* array;
	array=(int*)malloc(sizeof(int)*size);
	if ( !array ){
		fprintf(stderr,"\n unable to allocate int memory!\n");
		exit(-1);
	}
	else{
		for(int j=0;j<size;j++)
				array[j]=0.0;
	}
	
	return array;
}

real average(int count, ...)
{
    va_list ap;
    int j;
    real tot = 0;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(j = 0; j < count; j++)
        tot += va_arg(ap, real); /* Increments ap to the next argument. */
    va_end(ap);
    return tot / count;
}


void allocate_real(int Nx, int Ny, int Nz, int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	real** value= va_arg(ap, real**); /* Increments ap to the next argument. */
    	real* temp=allocate_d(Nx, Ny, Nz);
    	value[0]=temp;    	
    }
    va_end(ap);

}

void allocate_int(int Nx, int Ny, int Nz, int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	int** value= va_arg(ap, int**); /* Increments ap to the next argument. */
    	int* temp=allocate_i(Nx, Ny, Nz);
    	value[0]=temp;    	
    }
    va_end(ap);

}


void deallocate_real(int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	real* value= va_arg(ap, real*); /* Increments ap to the next argument. */
		free(value);
    }
    va_end(ap);

}
void deallocate_int(int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	int* value= va_arg(ap, int*); /* Increments ap to the next argument. */
		free(value);
    }
    va_end(ap);

}
