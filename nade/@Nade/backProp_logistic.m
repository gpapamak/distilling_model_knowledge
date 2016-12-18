function backProp_logistic(obj)
% Backpropagates derivatives through nade, with logistic hidden layer.
% Assumes that forward propagation has already been performed.
%
% George Papamakarios, Jun 2015

assert(obj.done_forwProp, 'Forward prop has to be run before backprop.');
N = size(obj.x, 2);

% derivatives wrt b
obj.dLdb = obj.x - obj.y;

% derivatives wrt U
obj.dLdU = repmat(permute(obj.dLdb, [1 3 2]), 1, obj.num_hidden, 1) .* obj.h;

% derivatives wrt a
dLda = obj.dLdU .* repmat(obj.U, 1, 1, N) .* (1 - obj.h);
dLda = cumsum(dLda(end:-1:1, :, :), 1);

% derivatives wrt c
obj.dLdc = dLda(end, :, :);

% derivatives wrt W
buf = repmat(permute(obj.x(1:end-1,:), [1 3 2]), 1, obj.num_hidden, 1);
obj.dLda = dLda(end-1:-1:1, :, :);
obj.dLdW = obj.dLda .* buf;

% set flags
obj.done_backProp = true;
