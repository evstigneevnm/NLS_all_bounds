#ifndef __FILE_OPERATIONS_H__
#define __FILE_OPERATIONS_H__

#include <stdio.h>
#include <stdlib.h>
#include "Macros.h"

void read_control_file(int N, real* x1_Re, real* x1_Im, real* x2_Re, real* x2_Im, real* x3_Re, real* x3_Im, real* x4_Re, real* x4_Im);
void write_control_file(int N, real* x1_Re, real* x1_Im, real* x2_Re, real* x2_Im, real* x3_Re, real* x3_Im, real* x4_Re, real* x4_Im);
void write_file(char* file_name, real *array, int Nx, int Ny, real dx, real dy);
void write_file_matrix(char *file_name, real *Matrix, int Nx, int Ny);

#endif
