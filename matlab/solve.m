ROSENBROCK3_a1 = 0.7970967740096232;
ROSENBROCK3_a2 = 0.5913813968007854;
ROSENBROCK3_a3 = 0.1347052663841181;

ROSENBROCK2_a1 = 1/4;
ROSENBROCK2_a2 = 1/3;

ROSENBROCK1_a1 = 1;

beta = 1.43;
delta_beta = 0.0;
lka = 0.0026;
kappa = 0.0028;
g = 0.0052;                     


L=1.0;
N=32;
tau = 0.01;

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
    iM1{j} = inv(E-tau*ROSENBROCK2_a1.*Gi);
    iM2{j} = inv(E-tau*ROSENBROCK2_a2.*Gi);
    %iM3{j} = inv(E-tau*ROSENBROCK3_a3.*Gi);
    
end;


x1=0.31379418035549456*ones(N,1)+1.0e-6*sin(randn(N,1));
x2=-0.34486184665357905*ones(N,1);
x3=0.67671457773023869*ones(N,1);
x4=-1.3908982860701657*ones(N,1);

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
T=100;
skip=round(1); %/tau
x_t=zeros(T/skip,1);
for t=1:T
    x_hat_new=rosenbrock_step( N, tau, G, iM1, iM2, iM3, x_hat);
    x_hat=x_hat_new;
    x=ifft(x_hat,'symmetric');
    if(mod(t,skip)==0)
        x_t(t,1)=x(N/2,1);
        plot(x_t,'.');
        drawnow;
    end;
    if(isnan(x))
        disp nans!
        break
    end
end;
hold off;