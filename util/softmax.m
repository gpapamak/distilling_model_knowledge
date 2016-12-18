function [y, dy, d2y] = softmax(x)
% [y, dy, d2y] = softmax(x)
% Softmax function, operates columnwise.
% INPUT
%       x     a DxN array
% OUTPUT
%       y     DxN softmax of each column of x
%       dy    DxDxN first derivative
%       d2y   DxDxDxN second derivative
%
% NOTE: for extreme values, the gradients can underflow to 0
%
% George Papamakarios, Mar 2015

[D, N] = size(x);

% subtract max(x) from all x to avoid overflow of exp(x)
x = x - ones(D, 1) * max(x, [], 1);
y = exp(x);
y = y ./ (ones(D, 1) * sum(y, 1));

% get derivatives
if nargout > 1
    yi = repmat(permute(y, [1 3 2]), 1, D, 1);
    yj = repmat(permute(y, [3 1 2]), D, 1, 1);
    dy = yj .* (repmat(eye(D), 1, 1, N) - yi);
end
if nargout > 2
    yk = repmat(permute(y, [4 3 1 2]), D, D, 1, 1);
    yj = repmat(permute(y, [4 1 3 2]), D, 1, D, 1);
    dyij = repmat(permute(dy, [1 2 4 3]), 1, 1, D, 1);
    dyik = repmat(permute(dy, [1 4 2 3]), 1, D, 1, 1);
    deltajk = repmat(permute(eye(D), [4 1 2 3]), D, 1, 1, N);
    d2y = dyik .* (deltajk - yj) - dyij .* yk;
end
