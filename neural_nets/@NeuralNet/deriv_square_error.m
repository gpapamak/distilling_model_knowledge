function [L, dLdp] = deriv_square_error(obj, x, y)
% Square error of derivatives loss function.
%     loss = 1/2I sum_i ||dfi/dx - dyi/dx||^2
% INPUTS
%       x     input locations
%       y     output values and derivatives (values are actually discarded)
% OUTPUTS
%       L     loss
%       dLdp  derivative of loss wrt network parameters
%
% George Papamakarios, May 2015

% unpack target derivatives
dydx = y(2:end, :, :);

% forward and back prop
obj.forwProp(x);
obj.backProp();

err = obj.layers{1}.dydx - dydx;
obj.RbackProp(err);

% calc loss
L = mean(sum(err .* err, 1), 2);
L = mean(L) / 2;

% calc derivatives of loss wrt parameters
dLdp = mean(mean(obj.Hpvx, 2), 3);
