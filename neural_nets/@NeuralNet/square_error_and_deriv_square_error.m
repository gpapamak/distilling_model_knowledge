function [L, dLdp] = square_error_and_deriv_square_error(obj, x, y, lambda)
% Square error regularized with square error of derivatives.
%     loss = 1/2I sum_i ||fi - yi||^2 + lambda/2I sum_i ||dfi/dx - dyi/dx||^2
% INPUTS
%       x     input locations
%       y     output values and derivatives
% OUTPUTS
%       L     loss
%       dLdp  derivative of loss wrt network parameters
%
% George Papamakarios, May 2015

% unpack target and derivatives
dydx = y(2:end, :, :);
y = permute(y(1, :, :), [2 3 1]);

% forward and back prop
obj.forwProp(x);
obj.backProp();

% square error of values
err1 = obj.layers{end}.x - y;

L1 = mean(err1 .* err1, 1);
L1 = mean(L1) / 2;

err1 = repmat(permute(err1, [3 1 2]), obj.num_params, 1, 1);
dL1dp = mean(mean(obj.dydp .* err1, 2), 3);

% square error of derivatives
err2 = obj.layers{1}.dydx - dydx;

L2 = mean(sum(err2 .* err2, 1), 2);
L2 = mean(L2) / 2;

obj.RbackProp(err2);
dL2dp = mean(mean(obj.Hpvx, 2), 3);

% combine the two losses
L = L1 + lambda * L2;
dLdp = dL1dp + lambda * dL2dp;
