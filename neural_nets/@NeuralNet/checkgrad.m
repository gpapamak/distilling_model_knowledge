function [err, err_p, err_x] = checkgrad(obj, Np, Nx, eps)
% Checks the derivatives of the network using finite differences.
% INPUTS
%       Np      number of sets of parameters to test for (optional)
%       Nx      number of sets of inputs to test for (optional)
%       eps     epsilon for finite differences (optional)
% OUTPUTS
%       err     maximum error encountered in all checks
%       err_p   maximum error in the derivatives of the parameters
%       err_x   maximum error in the derivatives of the inputs
%
% George Papamakarios, Feb 2015

if nargin < 4
    eps = 1.0e-5;
end
if nargin < 3
    Nx = 10;
end
if nargin < 2
    Np = 10;
end

% save actual parameters
params = obj.params;

% generate random inputs and parameters
ps = 5 * randn(obj.num_params, Np, obj.arraytype);
xs = 5 * randn(obj.num_inputs, Nx, obj.arraytype);

err_p = zeros(Nx, obj.num_outputs, obj.arraytype);
err_x = zeros(Np, obj.num_outputs, obj.arraytype);

for i = 1:obj.num_outputs
    
    for j = 1:Nx
        err_p(j,i) = checkgrad(ps, @(p) fp(obj, xs(:,j), p, i), [obj.num_params, 1], eps);
    end

    for j = 1:Np
        err_x(j,i) = checkgrad(xs, @(x) fx(obj, x, ps(:,j), i), [obj.num_inputs, 1], eps);
    end
end

% restore parameters
obj.setParamsFromVec(params, false);

% return maximum errors
err_p = max(err_p(:));
err_x = max(err_x(:));
err = max(err_p, err_x);

% clear memory
obj.clear();


function [yi, dyidp] = fp(obj, x, p, i)
% Returns the derivatives of the i-th output wrt parameters.

obj.setParamsFromVec(p, false);

obj.forwProp(x);
obj.backProp();
yi = obj.layers{end}.x(i);
dyidp = obj.dydp(:,i);


function [yi, dyidx] = fx(obj, x, p, i)
% Returns the derivatives of the i-th output wrt inputs.

obj.setParamsFromVec(p, false);

obj.forwProp(x);
obj.backProp_inputs();
yi = obj.layers{end}.x(i);
dyidx = obj.layers{1}.dydx(:,i);
