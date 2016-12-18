function [err] = checkloss(obj, Np, Nx, Nd, eps)
% Checks the derivatives of the loss function using finite differences.
% INPUTS
%       Np      number of sets of parameters to test for (optional)
%       Nx      number of datasets to test for (optional)
%       Nd      number of datapoints in a dataset (optional)
%       eps     epsilon for finite differences (optional)
% OUTPUTS
%       err     maximum error encountered in all checks
%
% George Papamakarios, May 2015

% assert that the network has a loss function
assert(~isempty(obj.loss_fun_hl), 'Network has no loss function to be checked.');

if nargin < 5
    eps = 1.0e-5;
end
if nargin < 4
    Nd = 10;
end
if nargin < 3
    Nx = 10;
end
if nargin < 2
    Np = 10;
end

% save actual parameters
params = obj.params;

% generate random parameters
ps = 2 * randn(obj.num_params, Np, obj.arraytype);

% set the sizes of x and y
sizx = [obj.num_inputs, Nd];
if obj.loss_fun_needs_derivs
    sizy = [1 + obj.num_inputs, obj.num_outputs, Nd];
else
    sizy = [obj.num_outputs, Nd];
end

err = zeros(Nx, 1, obj.arraytype);

for i = 1:Nx
    x = 5 * randn(sizx, obj.arraytype);
    y = 5 * randn(sizy, obj.arraytype);
    err(i) = checkgrad(ps, @(p) fp(obj, x, y, p), [obj.num_params, 1], eps);
end

% restore parameters
obj.setParamsFromVec(params, false);

% return maximum error
err = max(err(:));

% clear memory
obj.clear();


function [L, dLdp] = fp(obj, x, y, p)
% Returns the derivatives of the loss wrt parameters.

obj.setParamsFromVec(p, false);
[L, dLdp] = obj.eval_loss(x, y);
