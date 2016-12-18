function backProp(obj, dydx)
% Backpropagates derivatives through the network.
% INPUT
%     dydx     derivatives of the outputs to backpropagate (optional;
%              defaults to the identity matrix, which corresponds to the
%              output function being the identity)
%
% George Papamakarios, Feb 2015

assert(obj.done_forwProp, 'Forward prop has to be run before backprop.');

if nargin < 2
    dydx = repmat(eye(obj.num_outputs, obj.arraytype), [1 1 size(obj.layers{1}.x, 2)]);
end

num_f_out = size(dydx, 2);
N = size(dydx, 3);

if ~isequal(size(obj.dydp), [obj.num_params, num_f_out, N])
    obj.dydp = zeros(obj.num_params, num_f_out, N, obj.arraytype);
end
j = obj.num_params;

for l = obj.num_layers:-1:1

    % derivatives wrt biases
    i = j - obj.num_units(l+1);
    dydb = obj.layers{l+1}.backProp(dydx);
    obj.dydp(i+1:j, :, :) = dydb;
    j = i;
    
    % derivatives wrt weights
    i = j - obj.num_units(l+1) * obj.num_units(l);
    buf = permute(obj.layers{l}.x, [3 1 2]);
    buf = repmat(buf, obj.num_units(l+1), 1, 1);
    buf = reshape(buf, [], 1, N);
    buf = repmat(buf, 1, num_f_out, 1);
    obj.dydp(i+1:j, :, :) = repmat(dydb, obj.num_units(l), 1, 1) .* buf;
    j = i;

    % derivatives wrt inputs
    dydx = obj.weights{l}' * reshape(dydb, obj.num_units(l+1), []);
    dydx = reshape(dydx, obj.num_units(l), num_f_out, N);
end

obj.layers{1}.dydx = dydx;
obj.done_backProp = true;
