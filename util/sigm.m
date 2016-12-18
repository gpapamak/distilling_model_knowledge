function [y, dy, d2y] = sigm(x)
% [y, dy, d2y] = sigm(x)
% Logistic sigmoid function.
% INPUT
%       x     a real array
% OUTPUT
%       y     elementwise sigmoid of x
%       dy    first derivative at x
%       d2y   second derivative at x
%
% George Papamakarios, Nov 2014

% limit the input to avoid underflow
thres = 36;
x(x >  thres) =  thres;
x(x < -thres) = -thres;

y = 1 ./ (1 + exp(-x));

% get derivatives
if nargout > 1
    dy = y .* (1-y);
end
if nargout > 2
    d2y = dy .* (1-2*y);
end
