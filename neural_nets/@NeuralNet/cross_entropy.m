function [L, dLdp] = cross_entropy(obj, x, y)
% Cross entropy loss function
%      loss = -y logx - (1-y) log(1-x)
% INPUTS
%       x     input locations
%       y     output values
% OUTPUTS
%       L     loss
%       dLdp  derivative of loss wrt network parameters
%
% NOTE: it's meant to work only with nets with a single output, which is a
% probability (strictly between 0 and 1).
% 
% George Papamakarios, May 2015

% forward prop to compute loss
obj.forwProp(x);
L = -y .* log(obj.layers{end}.x) - (1-y) .* log(1-obj.layers{end}.x);
L = mean(L);

% backprop to compute derivatives
obj.backProp(permute(-y ./ obj.layers{end}.x + (1-y) ./ (1-obj.layers{end}.x), [1 3 2]));
dLdp = mean(obj.dydp, 3);
