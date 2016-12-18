function [err, err_p, err_x] = checkgrad(obj, Np, Nx, eps)
% Checks the derivatives of nade using finite differences.
% INPUTS
%       Np      number of sets of parameters to test for (optional)
%       Nx      number of sets of inputs to test for (optional)
%       eps     epsilon for finite differences (optional)
% OUTPUTS
%       err     maximum error encountered in all checks
%       err_p   maximum error in the derivatives of the parameters
%       err_x   maximum error in the derivatives of the inputs
%
% George Papamakarios, Jun 2015

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
W = obj.W;
c = obj.c;
U = obj.U;
b = obj.b;

% generate random inputs and parameters
ps = obj.precision(randn(obj.num_params, Np, obj.arraytype));
xs = obj.precision(randn(obj.num_inputs, Nx, obj.arraytype) > 0);

err_p = zeros(Nx, 1);
err_x = zeros(Np, 1);

for i = 1:Nx
    err_p(i) = checkgrad(ps, @(p) fp(obj, xs(:,i), p), [obj.num_params, 1], eps);
end

for i = 1:Np
    e1 = checkgrad(xs, @(x) fx_v1(obj, x, ps(:,i)), [obj.num_inputs, 1], eps);
    e2 = checkgrad(xs, @(x) fx_v2(obj, x, ps(:,i)), [obj.num_inputs, 1], eps);
    err_x(i) = max(e1, e2);
end

% restore parameters
obj.W = W;
obj.c = c;
obj.U = U;
obj.b = b;

% return maximum errors
err_p = max(err_p);
err_x = max(err_x);
err = max(err_p, err_x);

% clear memory
obj.clear();


function [L, dLdp] = fp(obj, x, p)
% Returns the derivatives of the output wrt parameters.

obj.setParamsFromVec(p);

obj.forwProp(x);
obj.backProp();
L = gather(obj.L);
dLdp = gather([obj.dLdW(:); obj.dLdc(:); obj.dLdU(:); obj.dLdb]);


function [L, dLdx] = fx_v1(obj, x, p)
% Returns the derivatives of the output wrt inputs. Uses eval, which in
% turn uses backprop only for inputs.

obj.setParamsFromVec(p);
[L, ~, dLdx] = obj.eval(x);
L = gather(L);
dLdx = gather(dLdx);


function [L, dLdx] = fx_v2(obj, x, p)
% Returns the derivatives of the output wrt inputs. Uses backprop, followed
% by reduced backprop only for inputs.

obj.setParamsFromVec(p);

obj.forwProp(x);
obj.backProp();
obj.backProp_inputs_reduced();
L = gather(obj.L);
dLdx = gather(obj.dLdx);
