#ifndef __GAUSS_SOLVE_H__
#define __GAUSS_SOLVE_H__


#include "constants.h"
#include "Macros.h"

__global__ void ker_gauss_elim(int batch_size, real *m_in, real *m_out)
{   
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (!(i < batch_size)) return;

        real m[matrix_size][extended_matrix_size];          //matrix column

        #pragma unroll
        for (int ii1 = 0;ii1 < matrix_size;++ii1) {
                #pragma unroll
                for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) 
                    m[ii1][ii3] = m_in[IMB(i,ii1,ii3)];
        }

        //forward step
        #pragma unroll
        for (int ii1 = 0;ii1 < matrix_size;++ii1) {
                real diag = m[ii1][ii1];
                #pragma unroll
                for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) {
                        m[ii1][ii3] /= diag;
                }

                #pragma unroll
                for (int ii2 = 0;ii2 < matrix_size;++ii2) {
                        if (ii2 <= ii1) continue;
                        real mul = m[ii2][ii1];
                        #pragma unroll
                        for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) {
                                m[ii2][ii3] -= mul*m[ii1][ii3];
                        }
                }
        }

        //backward step
        #pragma unroll
        for (int ii1 = matrix_size-1;ii1 >= 0;--ii1) {
                #pragma unroll
                for (int ii2 = matrix_size-1;ii2 >= 0;--ii2) {
                        if (ii2 >= ii1) continue;
                        real mul = m[ii2][ii1];
                        #pragma unroll
                        for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) {
                                m[ii2][ii3] -= mul*m[ii1][ii3];
                        }
                }
        }

        #pragma unroll
        for (int ii1 = 0;ii1 < matrix_size;++ii1) {
                #pragma unroll
                for (int ii3 = 0;ii3 < extended_matrix_size;++ii3) 
                    m_out[IMB(i,ii1,ii3)] = m[ii1][ii3];
        }
}


#endif