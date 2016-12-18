function [y] = logsumexp(x)
% [y] = logsumexp(x)
% Returns the log of the sum of the exps along columns of matrix x.
%
% George Papamakarios, Jan 2015

N = size(x, 1);
xmax = max(x, [], 1);
x = x - ones(N, 1) * xmax;
y = xmax + log(sum(exp(x), 1));
