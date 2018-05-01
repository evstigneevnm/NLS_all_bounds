function [ x_hat_new ] = rosenbrock_step( N, tau, G, iM1, iM2, iM3, x_hat )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

ROSENBROCK3_c21 = 1.058925354610082;
ROSENBROCK3_c31 = 0.5;
ROSENBROCK3_c32 = -0.3759391872875334;
ROSENBROCK3_b21 = 8.0/7.0;
ROSENBROCK3_b31 = 71.0/254.0;
ROSENBROCK3_b32 = 7.0/36.0;
ROSENBROCK3_w1 = 0.125;
ROSENBROCK3_w2 = 0.125;
ROSENBROCK3_w3 = 0.75;


ROSENBROCK2_w1 = 1/2;
ROSENBROCK2_w2 = 1/2;
ROSENBROCK2_b21 = 1;
ROSENBROCK2_c21 = 5/12;

ROSENBROCK1_w1 = 1;
    
    k1=zeros(N,4);
    k2=zeros(N,4);
    k3=zeros(N,4);
    f_x=nonlinear_function( x_hat );    
    for j = 1:N 
        k1(j,:)=(iM1{j}*(tau*(G{j}*x_hat(j,:)'+f_x(j,:)')))';
    end;
    
    xb_hat=x_hat+ROSENBROCK2_b21.*k1;
    xc_hat=x_hat+ROSENBROCK2_c21.*k1;
    f_x=nonlinear_function( xc_hat );    
    for j = 1:N 
        k2(j,:)=(iM2{j}*(tau*(G{j}*xb_hat(j,:)'+f_x(j,:)')))';
    end;

%    xb_hat=x_hat+ROSENBROCK3_b31.*k1+ROSENBROCK3_b32*k2;
%    xc_hat=x_hat+ROSENBROCK3_c31.*k1+ROSENBROCK3_c32*k2;
%    f_x=nonlinear_function( xc_hat );      
%    for j = 1:N 
%        k3(j,:)=(iM3{j}*(tau*(G{j}*xb_hat(j,:)'+f_x(j,:)')))';
%    end;    
   
    x_hat_new=x_hat+ROSENBROCK2_w1*k1+ROSENBROCK2_w2*k2;%+ROSENBROCK3_w3*k3;
end

