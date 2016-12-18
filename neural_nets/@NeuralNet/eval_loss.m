function [L, dLdp] = eval_loss(obj, x, y)
% Evaluates the loss function of the network and its derivatives.
% INPUTS
%       x     input locations
%       y     output values (and maybe derivatives)
% OUTPUTS
%       L     loss
%       dLdp  derivative of loss wrt network parameters
% 
% George Papamakarios, May 2015

% make the arrangements for the platform
if obj.gpu
    createArray = @gpuArray;
else
    createArray = @(x) x;
end

[L, dLdp] = obj.loss_fun_hl(createArray(x), createArray(y));
