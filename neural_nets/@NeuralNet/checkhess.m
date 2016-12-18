function [err, err_p, err_x] = checkhess(obj, Np, Nx, Nv, eps)
% Checks the hessian-vector product using finite differences.
% INPUTS
%       Np      number of sets of parameters to test for (optional)
%       Nx      number of sets of inputs to test for (optional)
%       Nv      number of vectors to test for (optional)
%       eps     epsilon for finite differences (optional)
% OUTPUTS
%       err     maximum error encountered in all checks
%       err_p   maximum error in Hp * v
%       err_x   maximum error in Hx * v
%
% George Papamakarios, Feb 2015

if nargin < 5
    eps = 1.0e-6;
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
params = obj.params;

% generate random parameters and inputs
ps = 5 * randn(obj.num_params, Np, obj.arraytype);
xs = 5 * randn(obj.num_inputs, Nx, obj.arraytype);

err_p = -inf;
err_x = -inf;

for p = ps

    obj.setParamsFromVec(p, false);

    for x = xs
        for vi = 1:Nv
            
            vs = 5 * randn(obj.num_inputs, obj.num_outputs, obj.arraytype);

            % evaluate Hessian x vector with R{backprop}
            obj.forwProp(x);
            obj.backProp();
            obj.RbackProp(vs);
            Hpvx = obj.Hpvx;
            Hxvx = obj.Hxvx;
            
            for i = 1:obj.num_outputs
            
                v = vs(:, i);
                
                % evaluate gradient at x + eps/2 v using backprop
                obj.forwProp(x + eps/2 * v);
                obj.backProp();
                dydp_plus = obj.dydp;
                dydx_plus = obj.layers{1}.dydx;

                % evaluate gradient at x - eps/2 v using backprop
                obj.forwProp(x - eps/2 * v);
                obj.backProp();
                dydp_minus = obj.dydp;
                dydx_minus = obj.layers{1}.dydx;
                
                e = norm(Hpvx(:,i) - (dydp_plus(:,i) - dydp_minus(:,i)) / eps);
                err_p = max(err_p, e);
                
                e = norm(Hxvx(:,i) - (dydx_plus(:,i) - dydx_minus(:,i)) / eps);
                err_x = max(err_x, e);
                
            end
        end
    end
end

err = max(err_p, err_x);

% restore parameters
obj.setParamsFromVec(params, false);

% clear memory
obj.clear();
