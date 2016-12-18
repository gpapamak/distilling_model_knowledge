function [L, dLdp] = square_error(obj, x, y)
% Square error loss function
%      loss = 1/2 ||f(x) - y||^2
% INPUTS
%       x     input locations
%       y     output values
% OUTPUTS
%       L     loss
%       dLdp  derivative of loss wrt network parameters
% 
% George Papamakarios, May 2015

% forward prop to compute loss
obj.forwProp(x);
err = obj.layers{end}.x - y;
L = sum(err .* err, 1);
L = mean(L) / 2;

% backprop to compute derivatives
obj.backProp(permute(err, [1 3 2]));
dLdp = mean(obj.dydp, 3);
