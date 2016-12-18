function backProp_inputs(obj, dydx)
% Backpropagates derivatives through the network, but does not compute nor 
% store derivatives wrt parameters, only wrt inputs.
% INPUT
%     dydx     derivatives of the outputs to backpropagate (optional;
%              defaults to the identity matrix, which corresponds to the
%              output function being the identity)
%
% George Papamakarios, Apr 2015

assert(obj.done_forwProp, 'Forward prop has to be run before backprop.');

if nargin < 2
    dydx = repmat(eye(obj.num_outputs, obj.arraytype), [1 1 size(obj.layers{1}.x, 2)]);
end

num_f_out = size(dydx, 2);
N = size(dydx, 3);

for l = obj.num_layers:-1:1

    % backpropagate through nonlinearity
    dydz = obj.layers{l+1}.backProp(dydx);
    
    % backpropagate through linearity
    dydx = obj.weights{l}' * reshape(dydz, obj.num_units(l+1), []);
    dydx = reshape(dydx, obj.num_units(l), num_f_out, N);
end

obj.layers{1}.dydx = dydx;
