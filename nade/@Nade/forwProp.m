function forwProp(obj, x)
% Forward propagation in nade for specified input.
% INPUT
%     x   columns are input locations
% 
% George Papamakarios, Jun 2015

N = size(x, 2);
obj.x = x;

% the last input is thrown away
buf = repmat(permute(x(1:end-1,:), [1 3 2]), 1, obj.num_hidden, 1);

% first layer
a = cumsum([repmat(obj.c, 1, 1, N); repmat(obj.W, 1, 1, N) .* buf], 1);
obj.h = obj.hidden_f(a);

% second layer
a = permute(sum(repmat(obj.U, 1, 1, N) .* obj.h, 2), [1 3 2]) + obj.b * obj.precision(ones(1, N, obj.arraytype));
obj.y = obj.sigm(a);
obj.L = sum(x .* log(obj.y) + (1-x) .* log(1-obj.y), 1);

% set flags
obj.done_forwProp = true;
obj.done_backProp = false;
