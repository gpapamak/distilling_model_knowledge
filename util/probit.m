function [y, dy, d2y] = probit(x)
% [y, dy, d2y] = probit(x)
% Probit function. Equals a gaussian cdf for mean 0 and variance 1.
% INPUT
%       x     a real array
% OUTPUT
%       y     elementwise probit of x
%       dy    first derivative at x
%       d2y   second derivative at x
%
% George Papamakarios, Jan 2015

% limit the input to avoid underflow
thres = 8;
x(x >  thres) =  thres;
x(x < -thres) = -thres;

y = 0.5 * erf(x / sqrt(2)) + 0.5;

% get derivatives
if nargout > 1
    dy = normpdf(x);
end
if nargout > 2
    d2y = -x .* dy;
end
