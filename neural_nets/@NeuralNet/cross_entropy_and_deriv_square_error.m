function [L, dLdp] = cross_entropy_and_deriv_square_error(obj, x, y, lambda)
% Cross entropy regularized by square error of derivatives. For the cross
% entropy term to be interpreted as such, it is assumed that both the
% targets and the net's outputs are the log of a probability distribution.
%     loss = -sum(exp(yi) * fi) + lambda/2ID sum_i ||dfi/dx - dyi/dx||^2
% INPUTS
%       x     input locations
%       y     output values and derivatives
% OUTPUTS
%       L     loss
%       dLdp  derivative of loss wrt network parameters
%
% George Papamakarios, May 2015

% unpack targets and their derivatives
dydx = y(2:end, :, :);
y = exp(permute(y(1, :, :), [2 3 1]));

% forward prop
obj.forwProp(x);

% dot product loss
L1 = -sum(y .* obj.layers{end}.x, 1);
L1 = mean(L1);

obj.backProp(permute(-y, [1 3 2]));
dL1dp = mean(obj.dydp, 3);

% deriv square error loss
obj.backProp();
err = obj.layers{1}.dydx - dydx;

L2 = mean(sum(err .* err, 1), 2);
L2 = mean(L2) / 2;

obj.RbackProp(err);
dL2dp = mean(mean(obj.Hpvx, 2), 3);

% combine losses
lambda = lambda / size(x, 1);
L = L1 + lambda * L2;
dLdp = dL1dp + lambda * dL2dp;
