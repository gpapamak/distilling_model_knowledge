function [L, dLdp] = avg_score_matching(obj, x, y)
% Score matching averaged wrt target probability.
%     loss = y/2 ||d/dx log f(x) - d/dx log y||^2 + (1-y)/2 ||d/dx log 1-f(x) - d/dx log 1-y||^2
% INPUTS
%       x     input locations
%       y     output values and derivatives
% OUTPUTS
%       L     loss
%       dLdp  derivative of loss wrt network parameters
%
% NOTE: it's meant to work only with nets with a single output which is a
% probability, with values strictly between 0 and 1.
% 
% George Papamakarios, May 2015

% unpack targets and their derivatives
y = permute(y, [1 3 2]);
dydx = y(2:end, :);
y = y(1, :);

% forward and back prop
obj.forwProp(x);
obj.backProp();

% get output and its derivatives
f = obj.layers{end}.x;
dfdp = permute(obj.dydp, [1 3 2]);
dfdx = permute(obj.layers{1}.dydx, [1 3 2]);

% calc derivatives of log f(x)
dlfdp = dfdp ./ (ones(size(dfdp, 1), 1, obj.arraytype) * f);
dlfdx = dfdx ./ (ones(size(dfdx, 1), 1, obj.arraytype) * f);
dlydx = dydx ./ (ones(size(dydx, 1), 1, obj.arraytype) * y);
diff_dlfdx_dlydx = dlfdx - dlydx;

% calc Hessian x diff for log f(x)
obj.RbackProp(permute(diff_dlfdx_dlydx, [1 3 2]));
Hf = permute(obj.Hpvx, [1 3 2]);
Hlf = Hf ./ (ones(size(Hf, 1), 1, obj.arraytype) * f) - dlfdp .* (ones(size(dlfdp, 1), 1, obj.arraytype) * sum(dlfdx .* diff_dlfdx_dlydx, 1));

% calc derivatives of log 1-f(x)
dl1mfdp = dfdp ./ (ones(size(dfdp, 1), 1, obj.arraytype) * (f-1));
dl1mfdx = dfdx ./ (ones(size(dfdx, 1), 1, obj.arraytype) * (f-1));
dl1mydx = dydx ./ (ones(size(dydx, 1), 1, obj.arraytype) * (y-1));
diff_dl1mfdx_dl1mydx = dl1mfdx - dl1mydx;

% calc Hessian x diff for log 1-f(x)
obj.RbackProp(permute(diff_dl1mfdx_dl1mydx, [1 3 2]));
H1mf = permute(obj.Hpvx, [1 3 2]);
Hl1mf = H1mf ./ (ones(size(H1mf, 1), 1, obj.arraytype) * (f-1)) - dl1mfdp .* (ones(size(dl1mfdp, 1), 1, obj.arraytype) * sum(dl1mfdx .* diff_dl1mfdx_dl1mydx, 1));

% calc loss
L = (y .* sum(diff_dlfdx_dlydx .^ 2, 1) + (1-y) .* sum(diff_dl1mfdx_dl1mydx .^ 2, 1)) / 2;
L = mean(L);

% calc derivatives of loss wrt parameters
dLdp = (ones(size(Hlf, 1), 1, obj.arraytype) * y) .* Hlf + (ones(size(Hl1mf, 1), 1, obj.arraytype) * (1-y)) .* Hl1mf;
dLdp = mean(dLdp, 2);
