function backProp_inputs_reduced(obj)
% Recuced back propagation for inputs only. Assumes that backrop for
% parameters has already been run.
%
% George Papamakarios, Jun 2015

assert(obj.done_backProp, 'Backprop has to be run before reduced backprop for inputs.');
N = size(obj.x, 2);

% derivatives wrt x
buf = repmat(obj.W, 1, 1, N);
dLdx = sum(obj.dLda .* buf, 2);
dLdx = [permute(dLdx, [1 3 2]); obj.precision(zeros(1, N, obj.arraytype))];
obj.dLdx = dLdx + log(obj.y ./ (1-obj.y));
