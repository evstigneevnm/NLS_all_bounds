function [ ret_F_hat ] = nonlinear_function( x_hat )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
    RefGamma = 3.5E-3;
    ImfGamma = 3.5E-4;
    x1=ifft(x_hat(:,1), 'symmetric');
    x2=ifft(x_hat(:,2), 'symmetric');
    x3=ifft(x_hat(:,3), 'symmetric');
    x4=ifft(x_hat(:,4), 'symmetric');
    N=size(x1,1);
    
    x1(round(2*N/3.0:N),1)=0;
    x2(round(2*N/3.0:N),1)=0;
    x3(round(2*N/3.0:N),1)=0;
    x4(round(2*N/3.0:N),1)=0;
    
    %xx=norm(x3,2)^2+norm(x4,2)^2;
    xx=x3.*x3+x4.*x4;
    
    F=[0 0 0 0;0 0 0 0;0 0 ImfGamma RefGamma;0 0 -RefGamma ImfGamma];
    ret_F=F*([x1 x2 x3 x4])';
    ret_F=ret_F';
%    ret_F(round(2*N/3.0:N),:)=0;
    

    ret_F_hat(:,1)=-fft(ret_F(:,1));
    ret_F_hat(:,2)=-fft(ret_F(:,2));
    ret_F_hat(:,3)=-fft(ret_F(:,3).*xx);
    ret_F_hat(:,4)=-fft(ret_F(:,4).*xx);   
    
end

