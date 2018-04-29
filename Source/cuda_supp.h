#ifndef __H_CUDA_SUPP_H__
#define __H_CUDA_SUPP_H__

#include <stdarg.h>
#include <stdio.h>
#include <cufft.h>
#include <iostream>
#include <string>
#include <initializer_list>

#include "Macros.h"

bool InitCUDA(int);
cudaComplex* device_allocate_complex(int Nx, int Ny, int Nz);
real* device_allocate_real(int Nx, int Ny, int Nz);
void device_allocate_real(int Nx, int Ny, int Nz, real *m_device);
void device_host_real_cpy(real* device, real* host, int Nx, int Ny, int Nz);
void device_host_real_cpy(real* device, real* host, int size);
void host_device_real_cpy(real* host, real* device, int Nx, int Ny, int Nz);
void host_device_real_cpy(real* host, real* device, int size);
void device_allocate_all_real(int Nx, int Ny, int Nz, int count, ...);
void device_deallocate_all_real(int count, ...);
void device_allocate_all_complex(int Nx, int Ny, int Nz, int count, ...);
void device_deallocate_all_complex(int count, ...);

template <class T>
void device_allocate_all_real2(int size, std::initializer_list<T> list )
{
    for( auto elem : list )
    {
        //printf("%d\n", elem);
        device_allocate_real(size, 1, 1, elem);
    }
}


template <class T>
void device_from_host_all_real_cpy(int size, std::initializer_list<T> list_device, std::initializer_list<T> list_host)
{
    if(list_host.size()!=list_device.size()){
        printf("\nError in device_from_host_all_real_cpy: unequal number of host->device arrays provided!\n");
        exit(-1);
    }

    printf("%i\n", list_host.size());

    for(int j=0;j<list_host.size();j++)
    {
        device_host_real_cpy(list_device.begin()[j], list_host.begin()[j], size);
    }
    
}


template <class T>
void host_from_device_all_real_cpy(int size, std::initializer_list<T> list_host, std::initializer_list<T> list_device)
{
    if(list_host.size()!=list_device.size()){
        printf("\nError in host_from_device_all_real_cpy: unequal number of host->device arrays provided!\n");
        exit(-1);
    }

    printf("%i\n", list_host.size());

    for(int j=0;j<list_host.size();j++)
    {
        host_device_real_cpy(list_host.begin()[j], list_device.begin()[j], size);
    }
    
}


template <class T>
void device_deallocate_all(std::initializer_list<T> list )
{
    for( auto elem : list )
    {
        cudaFree(elem);
    }
}



#endif
