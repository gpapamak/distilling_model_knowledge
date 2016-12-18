function RbackProp_relu(obj, vx)
% R{backprop} in nade, with relu hidden layer. Assumes that both forward
% and backward propagation have already been performed.
% INPUT
%     vx   vector with which to find the product with the Hessian
%          note that vx has the same length as x; it is implicitly
%          assumed that vp (the part of v corresponding to the
%          parameters) is zero
%
% George Papamakarios, Jun 2015

assert(obj.done_forwProp & obj.done_backProp, 'Forward prop and backprop have to be run before R{backprop}.');
N = size(vx, 2);

% -- forward pass

% the last input is thrown away
buf_vx = repmat(permute(vx(1:end-1,:), [1 3 2]), 1, obj.num_hidden, 1);

% first layer
Ra = cumsum([obj.precision(zeros(1, obj.num_hidden, N, obj.arraytype)); repmat(obj.W, 1, 1, N) .* buf_vx], 1);
dhda = double(obj.h > 0);
Rh = dhda .* Ra;

% second layer
Ra = permute(sum(repmat(obj.U, 1, 1, N) .* Rh, 2), [1 3 2]);
Ry = obj.y .* (1-obj.y) .* Ra;

% -- backward pass

% derivatives wrt b
obj.RdLdb = vx - Ry;
RdLdb = repmat(permute(obj.RdLdb, [1 3 2]), 1, obj.num_hidden, 1);
 dLdb = repmat(permute(obj. dLdb, [1 3 2]), 1, obj.num_hidden, 1);

% derivatives wrt U
obj.RdLdU = RdLdb .* obj.h + dLdb .* Rh;

% derivatives wrt a
RdLda = RdLdb .* repmat(obj.U, 1, 1, N) .* dhda;
RdLda = cumsum(RdLda(end:-1:1, :, :), 1);

% derivatives wrt c
obj.RdLdc = RdLda(end, :, :);

% derivatives wrt W
buf_x = repmat(permute(obj.x(1:end-1,:), [1 3 2]), 1, obj.num_hidden, 1);
obj.RdLdW = RdLda(end-1:-1:1, :, :) .* buf_x + obj.dLda .* buf_vx;
