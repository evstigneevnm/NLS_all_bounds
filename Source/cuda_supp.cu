#include "cuda_supp.h"


bool InitCUDA()
{

	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no compartable device found.\n");
		return false;
	}
	
	int deviceNumber=0;
	int deviceNumberTemp=0;
	
	if(count>1){
		

			
		for(i = 0; i < count; i++) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, i);
			printf( "#%i:	%s, pci-bus id:%i %i %i	\n", i, &deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
		}
			
		printf("Device number for it to use>>>\n",i);
				scanf("%i", &deviceNumberTemp);
   		deviceNumber=deviceNumberTemp;
	
	}
	else{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceNumber);
		printf( "#%i:	%s, pci-bus id:%i %i %i	\n", deviceNumber, &deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
		printf( "		using it...\n");	
	}

	cudaSetDevice(deviceNumber);
	
	return true;
}





cudaComplex* device_allocate_complex(int Nx, int Ny, int Nz){
	cudaComplex* m_device;
	int mem_size=sizeof(cudaComplex)*Nx*Ny*Nz;
	
    cudaError_t cuerr=cudaMalloc((void**)&m_device, mem_size);
	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot allocate device array because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    }  

    return m_device;	

}


real* device_allocate_real(int Nx, int Ny, int Nz){
	real* m_device;
	int mem_size=sizeof(real)*Nx*Ny*Nz;
	
    cudaError_t cuerr=cudaMalloc((void**)&m_device, mem_size);
	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot allocate device array because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    }  

    return m_device;	

}


void device_host_real_cpy(real* device, real* host, int Nx, int Ny, int Nz){
	int mem_size=sizeof(real)*Nx*Ny*Nz;
	cudaError_t cuerr=cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice);
   	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot copy real array from host to device because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 

}


void host_device_real_cpy(real* host, real* device, int Nx, int Ny, int Nz){
	int mem_size=sizeof(real)*Nx*Ny*Nz;
	cudaError_t cuerr=cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost);
 	if (cuerr != cudaSuccess)
    {
		printf("Cannot copy real array from device to host because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 
}


void device_allocate_all_real(int Nx, int Ny, int Nz, int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	real** value=va_arg(ap, real**); /* Increments ap to the next argument. */
    	real* temp=device_allocate_real(Nx, Ny, Nz);
    	value[0]=temp;    	
    }
    va_end(ap);

}

void device_allocate_all_complex(int Nx, int Ny, int Nz, int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	cudaComplex** value=va_arg(ap, cudaComplex**); /* Increments ap to the next argument. */
    	cudaComplex* temp=device_allocate_complex(Nx, Ny, Nz);
    	value[0]=temp;    	
    }
    va_end(ap);

}


void device_deallocate_all_real(int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	real* value=va_arg(ap, real*); /* Increments ap to the next argument. */
		cudaFree(value);
    }
    va_end(ap);

}

void device_deallocate_all_complex(int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	cudaComplex* value=va_arg(ap, cudaComplex*); /* Increments ap to the next argument. */
		cudaFree(value);
    }
    va_end(ap);

}

