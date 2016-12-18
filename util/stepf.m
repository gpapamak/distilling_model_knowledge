function [y] = stepf(x, e)
% [y] = stepf(x, e)
% e-parameterized step function: y = e for x < 0 and y = 1-e for x >= 0
%
% George Papamakarios, Jan 2015

if nargin < 2
    e = 0;
end

y = e + (1 - 2*e) * (x >= 0);
