#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h> //for timer
//support
#include "Macros.h"
#include "cuda_supp.h"
#include "file_operations.h"
#include "math_support.h"
#include "memory_operations.h"
//time steppers
#include "RK_time_step.h"
#include "Rosenbrock_time_step.h"

//Jacobian experimantal!
#include "Jacobian.h"
#include "Floquet.h"

//=================================================================
//=====Some validation initial conditions and exact solutions======
//=================================================================




void set_all_zero(int N, real* x1,real* x2,real* x3, real* x4){

    int j;

    FS
        x1[j]=0.0;
        x2[j]=0.0;  
        x3[j]=0.0;
        x4[j]=0.0;
    FE

}


void Initial_conditions(int N, real* x1,real* x2,real* x3, real* x4){

    int j;

    FS
         // x1[j]=0.31379418035549456+1e-6*sin(2*PI*j/N);
         // x2[j]=-0.34486184665357905;  
         // x3[j]=0.67671457773023869;
         // x4[j]=-1.3908982860701657;
        
        //x1[j]=1e-4*sin(2.0*j/N);
        //x2[j]=1e-8*sin(3.0*j/N);
        //x3[j]=1e-8*sin(4.0*j/N);
        //x4[j]=1e-8*sin(5.0*j/N);

         x1[j]=0.31379418035549456+1e-6*sin(2*PI*j/N);
         x2[j]=-0.34486184665357905;  
         x3[j]=0.67671457773023869;
         x4[j]=-0.3908982860701657;


    FE


}


int main (int argc, char *argv[])
{
    int j,N=32, M=17;
    int plan_N;


    if(argc!=10){
        //       0  1  2   3     4  5        6               7        8       9
        printf("%s boundary N L Timesteps g CF Period_Timesteps dt_Fraction Method\n boundary = (D,N,P); N - Fourier modes; L - length; g - parameter; CF - control file (yes - Y/ no - N); Period_Timestep - timesteps in one period; dt_Fraction - dt on Monodromy matrix integration=dt/dt_Fraction; Method (RK for Runge Kutta 4 or RB for Rosenbrock 3 semi-implicit method).\n",argv[0]);
        return 0;
    }
    char boundary=argv[1][0];
    std::string method_type=argv[9];

    if((boundary!='P')&&(boundary!='N')&&(boundary!='D')){
        printf("\nCorrect boundary type is not selected (D,N or P)! Don't know the basis functions!\n");
        return 0;
    }
    printf("boundary = %c, integration method = %s\n", boundary, method_type.c_str());
    
    int method=0;
    if(method_type=="RK"){
        method=1;
    }else if(method_type=="RB"){
        method=2;
    }else{
        printf("\n incorrect integration method provided!\n");
        return 0;
    }

    char *file_name="control_file.dat";
    N=atoi(argv[2]);        //data size for Domain
    if(boundary=='P'){
        M=(int)floor(N/2)+1;    //periodic data size for Image 
        plan_N=N;
        file_name="control_file_P.dat";
    }
    else if(boundary=='D'){
        M=N-1;                  //Diriclet data for Image 
        plan_N=2*N+2;
        file_name="control_file_D.dat";
    }
    else{
        if(N%2==0){
            M=N;                    //Neumann data for Image, if Domain is even
            plan_N=N;
        }
        else{
            M=2*N;                    //Neumann data for Image if Domain is odd
            plan_N=2*N;            
        }
        file_name="control_file_P.dat";
    }


    int timesteps=atoi(argv[4]);
    real L=atof(argv[3]);
    real g=atof(argv[5]);
    char CFile=argv[6][0];
    int timestepsMonodromy=atof(argv[7]);
    real dt_coefficient=atof(argv[8]);


    //number of timesteps for file to dump results.
    int timestep_file=30000;
    


    //check for cuda and select device!
    if(!InitCUDA(0)) {
        return 0;
    }

    int device_number=-1;
    cudaGetDevice(&device_number);
    printf("\nSelected device #%i\n", device_number);


    double *some_test;
    cudaError_t cuerr=cudaMalloc(&some_test,1000*sizeof(double));
    cudaFree(some_test);
    if(cuerr!=cudaSuccess){
        fprintf(stderr, "Cannot allocate device array because: %s\n",
        cudaGetErrorString(cuerr));
        return(0);
    }
    printf("\ncuda device selected!\n");
    


    //estimate timestep
    real dx=L/N;
    real dt=min2(0.07*dx,1.0e-1);
    //dt=0.1;
    printf("N=%i, M=%i, dx=%f, dt=%f\n",N,M,dx,dt);
    //host arrays for physical space
    real *x1,* x2, *x3, *x4;
    
    //host arrays for fourier space
    real *x1_Re,* x2_Re, *x3_Re, *x4_Re, *x1_Im,* x2_Im, *x3_Im, *x4_Im;


    //host arrays for result output
    real *x1_sp,* x2_sp, *x3_sp, *x4_sp;
    
    allocate_real(N, 1, 1, 4, &x1, &x2, &x3, &x4);
    //allocate_all_real2(N, {x1, x2, x3, x4});
    allocate_real(M, 1, 1, 8, &x1_Re, &x2_Re, &x3_Re, &x4_Re, &x1_Im, &x2_Im, &x3_Im, &x4_Im);
    allocate_real(N, timestep_file, 1, 4, &x1_sp, &x2_sp, &x3_sp, &x4_sp);

    //structure for timer estimation
    struct timeval start, end;

    set_all_zero(N,x1,x2,x3,x4);
    
    set_all_zero(N*timestep_file,x1_sp,x2_sp,x3_sp,x4_sp);

    //initial conditions in physical space
    //one can use initial conditions in fourier space, but i omit it for now.
    Initial_conditions(N, x1,x2,x3,x4);


    //do we read a control file?
    if(CFile=='Y'){
        printf("reading control file...");
        read_control_file(file_name, M, x1_Re, x1_Im, x2_Re, x2_Im, x3_Re, x3_Im, x4_Re, x4_Im);
        printf(" done\n");
    }



    //infact cuda starts here!!!
    

    //host arrays for working
    real *k_laplace; //1D wave numbers Laplace operator
    real *mask_2_3; //mask matrix for 2/3 dealiaing advection 
    allocate_real(M, 1, 1, 2, &k_laplace, &mask_2_3);
    
    //cuda arrays:
    real *x1_d, *x2_d, *x3_d, *x4_d;    //physical space variables
    real *x1_Re_d,* x2_Re_d, *x3_Re_d, *x4_Re_d, *x1_Im_d,* x2_Im_d, *x3_Im_d, *x4_Im_d; //device arrays for fourier space to translate control file
    real *x1_c, *x2_c, *x3_c, *x4_c;
    cudaComplex *x1_hat, *x2_hat, *x3_hat, *x4_hat;
    cudaComplex *x1_p, *x2_p, *x3_p, *x4_p;
    cudaComplex *RHS1_1, *RHS2_1, *RHS3_1, *RHS4_1;
    cudaComplex *RHS1_2, *RHS2_2, *RHS3_2, *RHS4_2;
    cudaComplex *RHS1_3, *RHS2_3, *RHS3_3, *RHS4_3;
    cudaComplex *RHS1_4, *RHS2_4, *RHS3_4, *RHS4_4;

    //cut wavenumbers and real space multiplication
    cudaComplex *x1_hat_cut, *x2_hat_cut, *x3_hat_cut, *x4_hat_cut;
    cudaComplex *x1_hat_mul, *x2_hat_mul, *x3_hat_mul, *x4_hat_mul;
    real *Q3_mul, *Q4_mul;
    cudaComplex *Q3_hat_mul, *Q4_hat_mul;
    //mask matrix for 2/3 dealiaing advection in device mem and wavenumbers
    real *mask_2_3_d;
    real *k_laplace_d;
    //matrices for Rosenbrock method
    real *M1, *M2, *M3;
    real *M1_d, *M2_d, *M3_d, *iM_d;
    cudaComplex *x1b_hat, *x2b_hat, *x3b_hat, *x4b_hat, *x1c_hat, *x2c_hat, *x3c_hat, *x4c_hat;

    device_allocate_all_real(N, 1, 1, 4, &x1_d, &x2_d, &x3_d, &x4_d);
    device_allocate_all_real2(N, {x1_d, x2_d, x3_d, x4_d});
    
    device_allocate_all_real(M, 1, 1, 8, &x1_Re_d, &x1_Im_d, &x2_Re_d, &x2_Im_d,&x3_Re_d, &x3_Im_d,&x4_Re_d, &x4_Im_d);
    mask_2_3_d=device_allocate_real(M,1,1);
    k_laplace_d=device_allocate_real(M,1,1);
    device_allocate_all_real(N, 1, 1, 6, &x1_c,&x2_c,&x3_c,&x4_c, &Q3_mul, &Q4_mul);
    device_allocate_all_complex(M, 1, 1, 30, &x1_hat,&x2_hat,&x3_hat,&x4_hat, &RHS1_1,&RHS2_1,&RHS3_1,&RHS4_1, &RHS1_2,&RHS2_2,&RHS3_2,&RHS4_2, &RHS1_3,&RHS2_3,&RHS3_3,&RHS4_3, &RHS1_4,&RHS2_4,&RHS3_4,&RHS4_4, &x1_hat_cut, &x2_hat_cut, &x3_hat_cut, &x4_hat_cut, &x1_hat_mul, &x2_hat_mul, &x3_hat_mul, &x4_hat_mul, &Q3_hat_mul, &Q4_hat_mul);
    device_allocate_all_complex(M, 1, 1, 4,&x1_p,&x2_p,&x3_p,&x4_p);



    //init cufft plan


    cufftHandle planR2C; 
    cufftResult result;
    result = cufftPlan1d(&planR2C, plan_N, RealToComplex,1);
    if (result != CUFFT_SUCCESS) { printf ("*CUFFT R2C MakePlan* failed\n"); return; }

    cufftHandle planC2R; 
    result = cufftPlan1d(&planC2R, plan_N, ComplexToReal,1);
    if (result != CUFFT_SUCCESS) { printf ("*CUFFT C2R MakePlan* failed\n"); return; }

//set up unchanging wave numbers
    build_Laplace_Wavenumbers(M, L, k_laplace, boundary);
//set up dealiazing mask matrix
    build_mask_matrix(M, L, mask_2_3, boundary);

//  copy all to device memory
    printf("HOST->DEVICE...");
    device_host_real_cpy(k_laplace_d, k_laplace, M, 1, 1);
//copy dealiazing mask matrix
    device_host_real_cpy(mask_2_3_d, mask_2_3, M, 1,1);

//copy working arrays
    device_host_real_cpy(x1_d, x1, N, 1, 1);
    device_host_real_cpy(x2_d, x2, N, 1, 1);
    device_host_real_cpy(x3_d, x3, N, 1, 1);
    device_host_real_cpy(x4_d, x4, N, 1, 1);
    device_host_real_cpy(x1_Re_d, x1_Re, M, 1, 1);
    device_host_real_cpy(x2_Re_d, x2_Re, M, 1, 1);
    device_host_real_cpy(x3_Re_d, x3_Re, M, 1, 1);
    device_host_real_cpy(x4_Re_d, x4_Re, M, 1, 1);  
    device_host_real_cpy(x1_Im_d, x1_Im, M, 1, 1);
    device_host_real_cpy(x2_Im_d, x2_Im, M, 1, 1);
    device_host_real_cpy(x3_Im_d, x3_Im, M, 1, 1);
    device_host_real_cpy(x4_Im_d, x4_Im, M, 1, 1);



//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //Set dim blocks for Device

    //Physical Domain 

    int k1=N/(BLOCKSIZE)+1;
    dim3 dimBlockD(BLOCKSIZE, 1);
    dim3 dimGridD( k1, 1 );

    // Fourier Image
    int k2=M/(BLOCKSIZE)+1;
    dim3 dimBlockI(BLOCKSIZE, 1);
    dim3 dimGridI( k2, 1 );


    printf("\n          BLOCK_DIM=%i ",BLOCKSIZE);
    printf("Domain dimGrid( %i,1,1 ), Fourier Image dimGrid( %i,1,1 ).\n",k1,k2);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//init Fourier modes
    Domain_To_Image(planR2C, x1_d, x1_hat, x2_d, x2_hat, x3_d, x3_hat, x4_d, x4_hat);


    if(CFile=='Y'){
        //set all fourier space from a control file
        all_real_2_complex(dimBlockI, dimGridI, x1_Re_d, x1_Im_d, x1_hat, x2_Re_d, x2_Im_d, x2_hat, x3_Re_d, x3_Im_d, x3_hat, x4_Re_d, x4_Im_d, x4_hat, M);
    }




    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }
    printf(" done\n");


    real CFL=1.0;
    real TotalTime=0.0;

    
    //droping data to a file    
    FILE *stream;
    stream=fopen("time_dependant.dat", "w" );

    //drop time dependant data
    real x1_loc, x2_loc, x3_loc, x4_loc;

    dt*=CFL; //temp with constant time step!!!

    int timestep_file_count=0;


    if(method==2)
        init_matrices_Rosenbrock(M, g, k_laplace, dt, M1, M2, M3, M1_d, M2_d, M3_d, iM_d, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat);

    
    check_gauss(dimGridI, dimBlockI, M, g,  k_laplace, dt, x1_hat, x2_hat, x3_hat, x4_hat);

    printf("\n===calculating===\n");
    gettimeofday(&start, NULL);
    for(int t=0;t<timesteps;t++){

        //real time step starts here
    
        if(method==1)
            RK4_single_step(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_hat, x1_p, x2_p, x3_p, x4_p, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, RHS1_3, RHS2_3, RHS3_3, RHS4_3, RHS1_4, RHS2_4, RHS3_4, RHS4_4);
        else
            RB3_single_step(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_hat, x1_p, x2_p, x3_p, x4_p, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, RHS1_3, RHS2_3, RHS3_3, RHS4_3, M1_d, M2_d, M3_d, iM_d);



    

        //real time step ends here
        

        //TODO: lame file operation =( Fix this in the future!

        Image_to_Domain(dimGridD, dimBlockD, planC2R, N, x1_d, x1_hat, x2_d, x2_hat, x3_d, x3_hat, x4_d, x4_hat);
        TotalTime+=dt;

        host_device_real_cpy(&x1_loc, &(x1_d[N/3]), 1, 1, 1);
        host_device_real_cpy(&x2_loc, &(x2_d[N/3]), 1, 1, 1);
        host_device_real_cpy(&x3_loc, &(x3_d[N/3]), 1, 1, 1);
        host_device_real_cpy(&x4_loc, &(x4_d[N/3]), 1, 1, 1);
        fprintf( stream, "%e %e %e %e %e\n", TotalTime, x1_loc, x2_loc, x3_loc, x4_loc);    


        //lame file operation ends here.

        if(t%217==0)
            printf("run:---[%.03f\%]---dt=%.03e---U=%.03e---\r",(t+1)/(1.0*timesteps)*100.0,dt,x1_loc);     
        

        if((timesteps-t)<timestep_file){

            host_device_real_cpy(x1, x1_d, N, 1, 1);
            host_device_real_cpy(x2, x2_d, N, 1, 1);
            host_device_real_cpy(x3, x3_d, N, 1, 1 );
            host_device_real_cpy(x4, x4_d, N, 1, 1 );
            for(int l=0;l<N;l++){
                x1_sp[N*timestep_file_count+l]=x1[l];
                x2_sp[N*timestep_file_count+l]=x2[l];
                x3_sp[N*timestep_file_count+l]=x3[l];
                x4_sp[N*timestep_file_count+l]=x4[l];
            }

            timestep_file_count++;
        }



    }
    gettimeofday(&end, NULL);
    fclose(stream);
    if(method==2)
         clean_matrices_Rosenbrock(M1, M2, M3, M1_d, M2_d, M3_d, iM_d, x1b_hat, x2b_hat, x3b_hat, x4b_hat, x1c_hat, x2c_hat, x3c_hat, x4c_hat);

    real etime=((end.tv_sec-start.tv_sec)*1000000u+(end.tv_usec-start.tv_usec))/1.0E6;
    printf("\n\nWall time:%fsec\n",etime);  
    
    //print Floquet Matrix 8102
    gettimeofday(&start, NULL);
    //CHANGE HERE!
    print_Floquet((int)timestepsMonodromy*dt_coefficient, dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt/dt_coefficient, g, x1_hat, x2_hat, x3_hat, x4_hat, x1_p, x2_p, x3_p, x4_p, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, RHS1_3, RHS2_3, RHS3_3, RHS4_3, RHS1_4, RHS2_4, RHS3_4, RHS4_4);
    gettimeofday(&end, NULL);
    etime=((end.tv_sec-start.tv_sec)*1000000u+(end.tv_usec-start.tv_usec))/1.0E6;
    printf("\n\nMonodromy wall time:%fsec\n",etime);    

    printf("done\n");

    printf("Jacobian construciton...\n");
    //CHANGE HERE!!!
    print_Jacobian(dimGridD, dimBlockD, dimGridI, dimBlockI, planR2C, planC2R, N, M, dt, g, x1_hat, x2_hat, x3_hat, x4_hat, x3_hat_cut, x4_hat_cut,  x3_c, x4_c, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul, mask_2_3_d, k_laplace_d, RHS1_1, RHS1_2, RHS1_3, RHS2_1, RHS2_2, RHS2_3, RHS3_1, RHS3_2, RHS3_3, RHS4_1, RHS4_2, RHS4_3);

    printf("done\n");

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }   

    printf("DEVICE->HOST...\n");
    

    host_device_real_cpy(x1, x1_d, N, 1, 1);
    host_device_real_cpy(x2, x2_d, N, 1, 1);
    host_device_real_cpy(x3, x3_d, N, 1, 1 );
    host_device_real_cpy(x4, x4_d, N, 1, 1 );
    host_device_real_cpy(mask_2_3, mask_2_3_d, M, 1, 1);
    all_complex_2_real(dimBlockI, dimGridI, x1_Re_d, x1_Im_d, x1_hat, x2_Re_d, x2_Im_d, x2_hat, x3_Re_d, x3_Im_d, x3_hat, x4_Re_d, x4_Im_d, x4_hat, M);



    host_device_real_cpy(x1_Re, x1_Re_d, M, 1, 1);
    host_device_real_cpy(x2_Re, x2_Re_d, M, 1, 1);
    host_device_real_cpy(x3_Re, x3_Re_d, M, 1, 1);
    host_device_real_cpy(x4_Re, x4_Re_d, M, 1, 1);  
    host_device_real_cpy(x1_Im, x1_Im_d, M, 1, 1);
    host_device_real_cpy(x2_Im, x2_Im_d, M, 1, 1);
    host_device_real_cpy(x3_Im, x3_Im_d, M, 1, 1);
    host_device_real_cpy(x4_Im, x4_Im_d, M, 1, 1);

    printf("done\n");
    
    //remove all CUDA device memory 
    printf("cleaning up DEVICE memory...\n");
    //clean cufft
    result = cufftDestroy(planR2C);
    if (result != CUFFT_SUCCESS) { printf ("CUFFT Destroy PlanR2C failed\n"); return; }
    result = cufftDestroy(planC2R);
    if (result != CUFFT_SUCCESS) { printf ("CUFFT Destroy PlanR2C failed\n"); return; }

    //clean all other arrays
    device_deallocate_all_real(14, x1_d, x2_d, x3_d, x4_d, x1_Re_d, x2_Re_d, x3_Re_d, x4_Re_d, x1_Im_d, x2_Im_d, x3_Im_d, x4_Im_d, mask_2_3_d, k_laplace_d);    
    
    device_deallocate_all_complex(36, x1_c, x2_c, x3_c, x4_c, x1_hat, x2_hat, x3_hat, x4_hat, RHS1_1, RHS2_1, RHS3_1, RHS4_1, RHS1_2, RHS2_2, RHS3_2, RHS4_2, RHS1_3, RHS2_3, RHS3_3, RHS4_3, RHS1_4, RHS2_4, RHS3_4, RHS4_4, x1_hat_cut, x2_hat_cut, x3_hat_cut, x4_hat_cut, x1_hat_mul, x2_hat_mul, x3_hat_mul, x4_hat_mul, Q3_mul, Q4_mul, Q3_hat_mul, Q4_hat_mul);

    device_deallocate_all_complex(4, x1_p, x2_p, x3_p, x4_p);

    printf("done\n");
    


    printf("wrighting file...\n");
    
    write_file("x1.dat", x1_sp, N, timestep_file, dx, dt);
    write_file("x2.dat", x2_sp, N, timestep_file, dx, dt);
    write_file("x3.dat", x3_sp, N, timestep_file, dx, dt);
    write_file("x4.dat", x4_sp, N, timestep_file, dx, dt);



    write_control_file(file_name, M, x1_Re, x1_Im, x2_Re, x2_Im, x3_Re, x3_Im, x4_Re, x4_Im);

    printf("done\n");
    printf("cleaning up HOST memory...\n"); 
//  clean all arrays



    deallocate_real(18, x1, x2, x3, x4, x1_Re, x2_Re, x3_Re, x4_Re, x1_Im, x2_Im, x3_Im, x4_Im, x1_sp, x2_sp, x3_sp, x4_sp, k_laplace, mask_2_3);

    printf("done\n");   
    printf("=============all done=============\n");
    return 0;
} 
