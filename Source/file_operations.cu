#include "file_operations.h"


void read_control_file(char *f_name, int N, real* x1_Re, real* x1_Im, real* x2_Re, real* x2_Im, real* x3_Re, real* x3_Im, real* x4_Re, real* x4_Im){
    double x1_Re_l, x1_Im_l, x2_Re_l, x2_Im_l, x3_Re_l, x3_Im_l, x4_Re_l, x4_Im_l;
    FILE *stream;
    stream=fopen(f_name, "r" );
    for(int j=0;j<N;j++){
            
                fscanf( stream, "%lf %lf %lf %lf %lf %lf %lf %lf",  &x1_Re_l, &x1_Im_l, &x2_Re_l, &x2_Im_l, &x3_Re_l, &x3_Im_l, &x4_Re_l, &x4_Im_l); 

                x1_Re[j]=x1_Re_l;
                x1_Im[j]=x1_Im_l;
                x2_Re[j]=x2_Re_l;
                x2_Im[j]=x2_Im_l;
                x3_Re[j]=x3_Re_l;
                x3_Im[j]=x3_Im_l;
                x4_Re[j]=x4_Re_l;
                x4_Im[j]=x4_Im_l;



    }
    fclose(stream);



}


void write_control_file(char *f_name, int N, real* x1_Re, real* x1_Im, real* x2_Re, real* x2_Im, real* x3_Re, real* x3_Im, real* x4_Re, real* x4_Im){
    double x1_Re_l, x1_Im_l, x2_Re_l, x2_Im_l, x3_Re_l, x3_Im_l, x4_Re_l, x4_Im_l;
    FILE *stream;
    stream=fopen(f_name, "w" );
    for(int j=0;j<N;j++){
        x1_Re_l=x1_Re[j];
        x1_Im_l=x1_Im[j];
        x2_Re_l=x2_Re[j];
        x2_Im_l=x2_Im[j];
        x3_Re_l=x3_Re[j];
        x3_Im_l=x3_Im[j];
        x4_Re_l=x4_Re[j];
        x4_Im_l=x4_Im[j];
        fprintf( stream, "%.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le\n", (double)x1_Re_l, (double)x1_Im_l, (double)x2_Re_l, (double)x2_Im_l, (double)x3_Re_l, (double)x3_Im_l, (double)x4_Re_l, (double)x4_Im_l); 
    }
    fclose(stream);
}




void write_file(char* file_name, real *array, int N, int timesteps, real dx, real dt){
    FILE *stream;
    stream=fopen(file_name, "w" );
    int index=0;
    real x,y;
    for(int k=0;k<timesteps;k++){
        for(int j=0;j<N;j++){       
            index=N*k+j;
            fprintf(stream, "%.016e ", array[index]);   
        }

        fprintf(stream, "\n");
    }
    
    fclose(stream);

}



void write_file_matrix(char *file_name, real *Matrix, int Nx, int Ny){
    FILE *stream;
    stream=fopen(file_name, "w" );
    double value;
    for(int j=0;j<Nx;j++){
        for(int k=0;k<Ny;k++){      
            value=(double)Matrix[I(k,j)]; //!!! fix the index!
            fprintf(stream, "%.016le ", value); 
        }
        fprintf(stream, "\n");
    }
    
    fclose(stream);

}