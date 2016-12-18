function [flag] = isint(x)
% [flag] = isint(x)
% Returns true if x is a whole number array.
%
% George Papamakarios, Jan 2015

flag = isreal(x) && all(mod(x(:), 1) == 0);
