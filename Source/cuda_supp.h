#ifndef __H_CUDA_SUPP_H__
#define __H_CUDA_SUPP_H__

#include <stdarg.h>
#include <stdio.h>
#include <cufft.h>
#include "Macros.h"

bool InitCUDA(void);
cudaComplex* device_allocate_complex(int Nx, int Ny, int Nz);
real* device_allocate_real(int Nx, int Ny, int Nz);
void device_host_real_cpy(real* device, real* host, int Nx, int Ny, int Nz);
void host_device_real_cpy(real* host, real* device, int Nx, int Ny, int Nz);
void device_allocate_all_real(int Nx, int Ny, int Nz, int count, ...);
void device_deallocate_all_real(int count, ...);
void device_allocate_all_complex(int Nx, int Ny, int Nz, int count, ...);
void device_deallocate_all_complex(int count, ...);


#endif
