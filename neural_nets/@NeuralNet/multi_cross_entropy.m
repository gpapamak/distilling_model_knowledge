function [L, dLdp] = multi_cross_entropy(obj, x, y)
% Multilabel cross entropy loss function
%      loss = -sum(yi log(xi))
% INPUTS
%       x     input locations
%       y     output values
% OUTPUTS
%       L     loss
%       dLdp  derivative of loss wrt network parameters
%
% NOTE: it's meant to work with nets whose output is a probability
% distribution, with values strictly between 0 and 1.
% 
% George Papamakarios, May 2015

% forward prop to compute loss
obj.forwProp(x);
L = -sum(y .* log(obj.layers{end}.x), 1);
L = mean(L);

% backprop to compute derivatives
obj.backProp(permute(-y ./ obj.layers{end}.x, [1 3 2]));
dLdp = mean(obj.dydp, 3);
