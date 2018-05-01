function [G] = construct_linear_matrices( beta, delta_beta, lka, kappa, g, p )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    D=p.*0.5/beta.*[0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0];
    B=[lka -delta_beta 0 kappa;delta_beta lka -kappa 0;0 kappa lka-g delta_beta;-kappa 0 -delta_beta lka-g];
    G=D-B;

    
end

