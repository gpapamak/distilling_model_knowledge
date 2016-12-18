function [err] = checkgrad(x, f, sizx, eps, times)
% [err] = checkgrad(x, f, sizx, eps, times)
% Checks gradients using finite differences.
% INPUTS
%       x       DxN matrix, its column vectors are the checking locations
%       f       handle of the function
%       sizx    size of input to f and df (optional; defaults to Dx1)
%       eps     difference in x (optional)
%       times   number of random directions to check (optional)
% OUTPUTS
%       err     maximum error encountered in all checks
%
% George Papamakarios, Nov 2014

[D, N] = size(x);

if nargin < 5
    times = D;
end
if nargin < 4
    eps = 1.0e-3;
end
if nargin < 3
    sizx = [D, 1];
end

assert(prod(sizx) == D, 'Sizes don''t match.');

err = zeros(times, N);

for n = 1:N
    for t = 1:times

        u = randn(sizx);
        u = u / norm(u(:));
        xn = reshape(x(:,n), sizx);
        [~, dfn] = f(xn);
        err(t, n) = (f(xn + (eps/2)*u) - f(xn - (eps/2)*u)) / eps - u(:)' * dfn(:);

    end
end

err = max(abs(err(:)));
