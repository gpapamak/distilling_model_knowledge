function [err] = checkhess(obj, Np, Nx, Nv, eps)
% Checks the hessian-vector product using finite differences.
% INPUTS
%       Np      number of sets of parameters to test for (optional)
%       Nx      number of sets of inputs to test for (optional)
%       Nv      number of vectors to test for (optional)
%       eps     epsilon for finite differences (optional)
% OUTPUTS
%       err     maximum error encountered in all checks
%
% George Papamakarios, Jun 2015

if nargin < 5
    eps = 1.0e-5;
end
if nargin < 4
    Nv = 10;
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

% generate random parameters and inputs
ps = 2 * obj.precision(randn(obj.num_params, Np, obj.arraytype));
xs = 2 * obj.precision(randn(obj.num_inputs, Nx, obj.arraytype));

err = -inf;

for p = ps

    obj.setParamsFromVec(p);

    for x = xs
        for i = 1:Nv
            
            v = 2 * obj.precision(randn(obj.num_inputs, 1, obj.arraytype));

            % evaluate Hessian x vector with R{backprop}
            obj.forwProp(x);
            obj.backProp();
            obj.RbackProp(v);
            RdLdW = obj.RdLdW(:);
            RdLdc = obj.RdLdc(:);
            RdLdU = obj.RdLdU(:);
            RdLdb = obj.RdLdb(:);
            
            % evaluate gradient at x + eps/2 v using backprop
            obj.forwProp(x + eps/2 * v);
            obj.backProp();
            dLdW_plus = obj.dLdW(:);
            dLdc_plus = obj.dLdc(:);
            dLdU_plus = obj.dLdU(:);
            dLdb_plus = obj.dLdb(:);

            % evaluate gradient at x - eps/2 v using backprop
            obj.forwProp(x - eps/2 * v);
            obj.backProp();
            dLdW_minus = obj.dLdW(:);
            dLdc_minus = obj.dLdc(:);
            dLdU_minus = obj.dLdU(:);
            dLdb_minus = obj.dLdb(:);

            e = zeros(1,4);
            e(1) = norm(RdLdW - (dLdW_plus - dLdW_minus) / eps);
            e(2) = norm(RdLdc - (dLdc_plus - dLdc_minus) / eps);
            e(3) = norm(RdLdU - (dLdU_plus - dLdU_minus) / eps);
            e(4) = norm(RdLdb - (dLdb_plus - dLdb_minus) / eps);
            err = max([err, e]);
        end
    end
end

% restore parameters
obj.W = W;
obj.c = c;
obj.U = U;
obj.b = b;

% clear memory
obj.clear();
