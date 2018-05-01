beta = 1.43;
delta_beta = 0.0;
lka = 0.0026;
kappa = 0.0028;
g = 0.0252;                     


L=10.0;
N=32;
tau = 0.05;

grad_x = ((2*pi/L).*[0:((N)/2-1) (-(N)/2):(-1)])';
laplace_x=-grad_x.^2;
E=eye(4,4);
G = cell(1,N);
iM1 = cell(1,N);
iM2 = cell(1,N);
iM3 = cell(1,N);
for j = 1:N
    Gi = construct_linear_matrices( beta, delta_beta, lka, kappa, g, laplace_x(j,1) );
    G{j} = Gi;
    iM1{j} = inv(E-0.5*tau.*Gi)*(E+0.5*tau.*Gi);
end;


%x1=0.31379418035549456*ones(N,1)+1.0e-6*randn(N,1);
%x2=-0.34486184665357905*ones(N,1);
%x3=0.67671457773023869*ones(N,1);
%x4=-1.3908982860701657*ones(N,1);

max_eig=0.0;
max_eig_j=0;
max_eig_no=0;
for j = 1:N
    max_eig_1=max(abs(eig(iM1{j})));
    max_eig_2=max(abs(eig(iM2{j})));
    max_eig_3=max(abs(eig(iM3{j})));
    if(max_eig<max_eig_1)
        max_eig_no=1;
        max_eig_j=j;
        max_eig=max_eig_1;
    end;
    if(max_eig<max_eig_2)
        max_eig_no=2;
        max_eig_j=j;
        max_eig=max_eig_2;
    end;
    if(max_eig<max_eig_3)
        max_eig_no=3;
        max_eig_j=j;
        max_eig=max_eig_3;
    end;
    
end;
max_eig
max_eig_no
max_eig_j
%str_e = sprintf('max abs eig=%e at matrix no=%i, index j=%i',max_eig, max_eig_no, max_eig_j)


x1_hat=fft(x1);
x2_hat=fft(x2);
x3_hat=fft(x3);
x4_hat=fft(x4);
x_hat=[x1_hat x2_hat x3_hat x4_hat];

hold on;
T=150000;
Tstore=1000;
skip=100;
x_t=zeros(T,1);
x1t_sapce=zeros(N,Tstore);
x_hat_new=x_hat;
for t=1:T
    
    for j=1:N
        x_hat_new(j,:)=iM1{j}*x_hat(j,:)';
    end
    x_hat0=x_hat_new;
    
    f_x=nonlinear_function( x_hat_new ); 
    for j=1:N
        x_hat(j,:)=x_hat0(j,:)+tau*f_x(j,:);
    end
    
    f_x=nonlinear_function( x_hat);
    for j=1:N
        x_hat_new(j,:)=0.75*x_hat0(j,:)+0.25*(x_hat(j,:)+tau*f_x(j,:));
    end
    
    f_x=nonlinear_function( x_hat_new);
    for j=1:N
        x_hat(j,:)=1/3*x_hat0(j,:)+2/3*(x_hat_new(j,:)+tau*f_x(j,:));
    end
    
    
    x=ifft(x_hat,'symmetric');
    
    x_t(t,1)=x(N/2,1);
    
    if((T-Tstore)<t)
        x1t_sapce(:,t-(T-Tstore))=x1(:,1);
    end
    
    if(mod(t,skip)==0)
        %plot(x_t,'.');
        %drawnow;
    end;
    if(isnan(x))
        disp nans!
        break
    end
end;
hold off;