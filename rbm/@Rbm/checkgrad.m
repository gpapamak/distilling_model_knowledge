function [err] = checkgrad(obj, Np, Nx, eps)
% Checks the derivatives of log p(x) wrt to x using finite differences.
% INPUTS
%       Np      number of sets of parameters to test for (optional)
%       Nx      number of sets of inputs to test for (optional)
%       eps     epsilon for finite differences (optional)
% OUTPUTS
%       err     maximum error encountered in all checks
%
% George Papamakarios, Jun 2015

if nargin < 4
    eps = 1.0e-5;
end
if nargin < 3
    Nx = 20;
end
if nargin < 2
    Np = 20;
end

% save actual parameters
W = obj.W;
a = obj.a;
b = obj.b;

% generate random inputs
xs = double(randn(obj.num_inputs, Nx, obj.arraytype) > 0);

err = zeros(Np, 1);

for i = 1:Np
    
    obj.W = 5 * randn(size(W), obj.arraytype);
    obj.a = 5 * randn(size(a), obj.arraytype);
    obj.b = 5 * randn(size(b), obj.arraytype);
    
    err(i) = checkgrad(xs, @obj.eval, [obj.num_inputs, 1], eps);
end

% restore parameters
obj.W = W;
obj.a = a;
obj.b = b;

% return maximum error
err = max(err);
