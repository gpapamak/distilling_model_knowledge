function [L, dLdp] = dot_product(obj, x, y)
% Dot product loss function
%      loss = -sum(yi * xi)
% Note that with logsoftmax final layer, it is the same as multi cross
% entropy.
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
L = -sum(y .* obj.layers{end}.x, 1);
L = mean(L);

% backprop to compute derivatives
obj.backProp(permute(-y, [1 3 2]));
dLdp = mean(obj.dydp, 3);
