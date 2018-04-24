file:
	2>result_make.txt
	nvcc -g -m64  Source/file_operations.cu -o Obj/file_operations.o -c 2>>result_make.txt

supp:
	2>result_make.txt
	nvcc -g -m64  Source/cuda_supp.cu -o Obj/cuda_supp.o -c 2>>result_make.txt
	nvcc -g -m64  Source/file_operations.cu -o Obj/file_operations.o -c 2>>result_make.txt
	g++ -g Source/memory_operations.c -o Obj/memory_operations.o -c 2>>result_make.txt
	
adv_2p3:
	2>result_make.txt
	nvcc -g -m64  Source/advection_2_3.cu -o Obj/advection_2_3.o -c 2>>result_make.txt


math:
	2>result_make.txt
	nvcc -g -m64  Source/math_support.cu -o Obj/math_support.o -c 2>>result_make.txt


rkstep:
	2>result_make.txt
	nvcc -g -m64  Source/RK_time_step.cu -o Obj/RK_time_step.o -c 2>>result_make.txt


deb:
	2>result_make.txt
	nvcc -g -m64  Source/NLS_periodic.cu -o NLS_periodic -lcufft -lm Obj/cuda_supp.o Obj/file_operations.o  Obj/math_support.o Obj/memory_operations.o Obj/advection_2_3.o Obj/RK_time_step.o  2>>result_make.txt


debJacobian:
	2>result_make.txt
	nvcc -g -m64  Source/NLS_periodic.cu -o NLS_periodic -lcufft -lm Obj/cuda_supp.o Obj/file_operations.o  Obj/math_support.o Obj/memory_operations.o Obj/advection_2_3.o Obj/RK_time_step.o Obj/Jacobian.o 2>>result_make.txt

debFloquet:
	2>result_make.txt
	nvcc -g -m64  Source/NLS_periodic.cu -o NLS_periodic -lcufft -lm -lcublas Obj/cuda_supp.o Obj/file_operations.o  Obj/math_support.o Obj/memory_operations.o Obj/advection_2_3.o Obj/RK_time_step.o Obj/Jacobian.o Obj/Floquet.o 2>>result_make.txt

Jacobian:
	2>result_make.txt
	nvcc -g -m64  Source/Jacobian.cu -o Obj/Jacobian.o -c 2>>result_make.txt

Floquet:
	2>result_make.txt
	nvcc -g -m64  Source/Floquet.cu -o Obj/Floquet.o -c 2>>result_make.txt

all:
	make adv_2p3 supp math rkstep Jacobian Floquet debFloquet

clear:
	cd Obj; \
	rm *.o 	
