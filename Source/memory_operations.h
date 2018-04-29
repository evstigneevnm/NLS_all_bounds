#ifndef __MEMORY_OPERATIONS_H__
#define __MEMORY_OPERATIONS_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <iostream>
#include <string>
#include <initializer_list>
#include "Macros.h"

void allocate_d(int size, real **array);
real* allocate_d(int Nx, int Ny, int Nz);
int* allocate_i(int Nx, int Ny, int Nz);
void allocate_real(int Nx, int Ny, int Nz, int count, ...);
void allocate_int(int Nx, int Ny, int Nz, int count, ...);
void deallocate_real(int count, ...);
void deallocate_int(int count, ...);

template <class T>
void allocate_all_real2(int size, std::initializer_list<T> list )
{
    for( auto elem : list )
    {
        //printf("%d\n", elem);
        allocate_d(size, &elem);
    }
}


#endif
