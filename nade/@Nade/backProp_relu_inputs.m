function backProp_relu_inputs(obj)
% Backpropagates derivatives through nade, with relu hidden layer. It only
% computes and stores the derivatives wrt the input. Assumes that forward
% propagation has already been performed.
%
% George Papamakarios, Jun 2015

assert(obj.done_forwProp, 'Forward prop has to be run before backprop.');
N = size(obj.x, 2);

% derivatives wrt b
dLdb = obj.x - obj.y;
dLdb = repmat(permute(dLdb, [1 3 2]), 1, obj.num_hidden, 1);

% derivatives wrt a
dLda = dLdb .* repmat(obj.U, 1, 1, N) .* double(obj.h > 0);
dLda = cumsum(dLda(end:-1:2, :, :), 1);

% derivatives wrt x
buf = repmat(obj.W, 1, 1, N);
dLdx = sum(dLda(end:-1:1, :, :) .* buf, 2);
dLdx = [permute(dLdx, [1 3 2]); obj.precision(zeros(1, N, obj.arraytype))];
obj.dLdx = dLdx + log(obj.y ./ (1-obj.y));
