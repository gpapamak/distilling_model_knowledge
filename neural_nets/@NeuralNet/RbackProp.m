function RbackProp(obj, vx, d2ydx2)
% Computes the product of the Hessian of the network with some vector.
% INPUT
%     vx       vector with which to find the product with the Hessian
%              note that vx has the same length as x; it is implicitly
%              assumed that vp (the part of v corresponding to the
%              parameters) is zero
%     d2ydx2   second derivatives with respect to the outputs (optional;
%              defaults to all zeros, which corresponds to the output
%              function being the identity)
%
% George Papamakarios, Feb 2015

assert(obj.done_forwProp & obj.done_backProp, 'Forward prop and backprop have to be run before R{backprop}.');

num_f_out = size(vx, 2);
N = size(vx, 3);

% -- forward pass

obj.layers{1}.Rx = vx;

for l = 1:obj.num_layers
    
    Rz = obj.weights{l} * reshape(obj.layers{l}.Rx, obj.num_units(l), []);
    Rz = reshape(Rz, obj.num_units(l+1), num_f_out, N);
    obj.layers{l+1}.RforwProp(Rz);
end

% -- backward pass

if nargin < 3
    obj.Hxvx = zeros(obj.num_outputs, num_f_out, N, obj.arraytype);
else
    % NOTE: this calculation is n^3 in the number of outputs;
    % if the number of outputs is small (which is usually the case) it
    % doesn't really matter, however it's possible to make it more
    % efficient by providing the hessian x vector product directly
    buf = repmat(permute(obj.layers{end}.Rx, [1 4 2 3]), 1, obj.num_outputs, 1, 1);
    obj.Hxvx = permute(sum(d2ydx2 .* buf, 1), [2 3 4 1]);
end

if ~isequal(size(obj.Hpvx), [obj.num_params, num_f_out, N])
    obj.Hpvx = zeros(obj.num_params, num_f_out, N, obj.arraytype);
end
k = obj.num_params;

for l = obj.num_layers:-1:1

    % derivatives wrt biases
    j = k - obj.num_units(l+1);
    Rdydb = obj.layers{l+1}.RbackProp(obj.Hxvx);
    obj.Hpvx(j+1:k, :, :) = Rdydb;

    % derivatives wrt weights
    buf1 = permute(obj.layers{l}.x, [3 1 2]);
    buf1 = repmat(buf1, obj.num_units(l+1), 1, 1);
    buf1 = reshape(buf1, [], 1, N);
    buf1 = repmat(buf1, 1, num_f_out, 1);
    
    buf2 = permute(obj.layers{l}.Rx, [4 1 2 3]);
    buf2 = repmat(buf2, obj.num_units(l+1), 1, 1, 1);
    buf2 = reshape(buf2, [], num_f_out, N);
    
    i = j - obj.num_units(l+1) * obj.num_units(l);
    obj.Hpvx(i+1:j, :, :) = repmat(Rdydb, obj.num_units(l), 1, 1) .* buf1 + repmat(obj.dydp(j+1:k,:,:), obj.num_units(l), 1, 1) .* buf2;
    k = i;

    % derivatives wrt inputs
    obj.Hxvx = obj.weights{l}' * reshape(Rdydb, obj.num_units(l+1), []);
    obj.Hxvx = reshape(obj.Hxvx, obj.num_units(l), num_f_out, N);
end
