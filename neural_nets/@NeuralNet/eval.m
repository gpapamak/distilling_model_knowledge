function [y, dydx] = eval(obj, x)
% Evaluates the neural network at specified input locations.
% INPUT
%     x     columns are input locations
% OUTPUT
%     y     network's output at x
%     dydx  y's derivatives with respect to x's
% 
% George Papamakarios, Feb 2015

% make the arrangements for the platform
if obj.gpu
    createArray = @gpuArray;
else
    createArray = @(x) x;
end

obj.forwProp(createArray(x));
y = obj.layers{end}.x;

if nargout > 1
    obj.backProp_inputs();
    dydx = obj.layers{1}.dydx;
end

obj.clear();
